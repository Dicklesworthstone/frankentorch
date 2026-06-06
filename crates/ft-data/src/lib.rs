#![forbid(unsafe_code)]

use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, TensorNodeId};

fn checked_shape_numel(shape: &[usize], reason: &'static str) -> Result<usize, AutogradError> {
    if shape.is_empty() {
        return Ok(1);
    }
    let mut product = 1usize;
    for &dim in shape {
        if dim == 0 {
            return Ok(0);
        }
        product = product.checked_mul(dim).ok_or(AutogradError::Dispatch(
            ft_dispatch::DispatchError::Key(ft_dispatch::DispatchKeyError::IncompatibleSet {
                reason,
            }),
        ))?;
    }
    Ok(product)
}

fn checked_mul(lhs: usize, rhs: usize, reason: &'static str) -> Result<usize, AutogradError> {
    lhs.checked_mul(rhs)
        .ok_or(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
            ft_dispatch::DispatchKeyError::IncompatibleSet { reason },
        )))
}

fn transform_config_error(reason: &'static str) -> AutogradError {
    AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
        ft_dispatch::DispatchKeyError::IncompatibleSet { reason },
    ))
}

fn dataloader_error(reason: &'static str) -> AutogradError {
    AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
        ft_dispatch::DispatchKeyError::IncompatibleSet { reason },
    ))
}

// ── Data Item ────────────────────────────────────────────────────────────

/// A single data sample returned by a `Dataset`.
///
/// Each item consists of named tensors (e.g., "input" and "target").
/// The tensors are represented as flat f64 vectors with associated shapes.
#[derive(Debug, Clone)]
pub struct DataItem {
    /// Named tensor data: (name, values, shape).
    pub tensors: Vec<(String, Vec<f64>, Vec<usize>)>,
}

impl DataItem {
    /// Create a data item with a single input-target pair.
    pub fn input_target(
        input_values: Vec<f64>,
        input_shape: Vec<usize>,
        target_values: Vec<f64>,
        target_shape: Vec<usize>,
    ) -> Self {
        Self {
            tensors: vec![
                ("input".to_string(), input_values, input_shape),
                ("target".to_string(), target_values, target_shape),
            ],
        }
    }

    /// Create a data item with a single tensor.
    pub fn single(name: &str, values: Vec<f64>, shape: Vec<usize>) -> Self {
        Self {
            tensors: vec![(name.to_string(), values, shape)],
        }
    }
}

// ── Dataset Trait ────────────────────────────────────────────────────────

/// Trait for datasets that provide indexed access to data samples.
pub trait Dataset {
    /// The total number of samples in the dataset.
    fn len(&self) -> usize;

    /// Whether the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the data item at the given index.
    ///
    /// # Panics
    /// May panic if `index >= self.len()`.
    fn get(&self, index: usize) -> DataItem;

    fn collate_indices(
        &self,
        _indices: &[usize],
        _session: &mut FrankenTorchSession,
    ) -> Option<Result<Batch, AutogradError>> {
        None
    }
}

// ── TensorDataset ────────────────────────────────────────────────────────

/// A dataset backed by a collection of pre-loaded tensor data.
///
/// Each sample is a `DataItem` containing one or more named tensors.
/// This is equivalent to PyTorch's `TensorDataset`.
#[derive(Clone)]
pub struct TensorDataset {
    items: Vec<DataItem>,
    columns: Option<Vec<TensorDatasetColumn>>,
}

#[derive(Clone)]
struct TensorDatasetColumn {
    name: String,
    shape: Vec<usize>,
    sample_numel: usize,
    values: Vec<f64>,
}

impl TensorDataset {
    /// Create a new `TensorDataset` from a vector of data items.
    pub fn new(items: Vec<DataItem>) -> Self {
        let columns = build_tensor_dataset_columns(&items);
        Self { items, columns }
    }

    /// Create from parallel vectors of inputs and targets.
    ///
    /// Each input/target pair becomes one `DataItem` with "input" and "target" tensors.
    pub fn from_inputs_targets(
        inputs: Vec<(Vec<f64>, Vec<usize>)>,
        targets: Vec<(Vec<f64>, Vec<usize>)>,
    ) -> Self {
        assert_eq!(
            inputs.len(),
            targets.len(),
            "inputs and targets must have same length"
        );
        let items: Vec<DataItem> = inputs
            .into_iter()
            .zip(targets)
            .map(|((iv, is), (tv, ts))| DataItem::input_target(iv, is, tv, ts))
            .collect();
        let columns = build_tensor_dataset_columns(&items);
        Self { items, columns }
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, index: usize) -> DataItem {
        self.items[index].clone()
    }

    fn collate_indices(
        &self,
        indices: &[usize],
        session: &mut FrankenTorchSession,
    ) -> Option<Result<Batch, AutogradError>> {
        Some(if let Some(columns) = &self.columns {
            collate_tensor_dataset_columns(session, columns, self.items.len(), indices)
        } else {
            collate_tensor_dataset_items(session, &self.items, indices)
        })
    }
}

fn build_tensor_dataset_columns(items: &[DataItem]) -> Option<Vec<TensorDatasetColumn>> {
    let first = items.first()?;
    let mut columns = Vec::with_capacity(first.tensors.len());
    for (name, values, shape) in &first.tensors {
        let sample_numel =
            checked_shape_numel(shape, "DataLoader: sample shape volume overflow").ok()?;
        if values.len() != sample_numel {
            return None;
        }
        columns.push(TensorDatasetColumn {
            name: name.clone(),
            shape: shape.clone(),
            sample_numel,
            values: Vec::with_capacity(items.len().checked_mul(sample_numel)?),
        });
    }

    for item in items {
        if item.tensors.len() != columns.len() {
            return None;
        }
        for ((name, values, shape), column) in item.tensors.iter().zip(columns.iter_mut()) {
            let same_name = name.as_str().cmp(column.name.as_str()).is_eq();
            let same_shape = shape.as_slice().cmp(column.shape.as_slice()).is_eq();
            if !same_name || !same_shape || values.len() != column.sample_numel {
                return None;
            }
            column.values.extend_from_slice(values);
        }
    }

    Some(columns)
}

// ── Batch ────────────────────────────────────────────────────────────────

/// A collated batch of data items, ready for forward pass.
///
/// Contains named tensors that have been stacked along a new batch dimension.
/// For example, if individual samples have shape `[features]`, the batch
/// tensor has shape `[batch_size, features]`.
pub struct Batch {
    /// Named batched tensors as `TensorNodeId`s registered in a session.
    pub tensors: Vec<(String, TensorNodeId)>,
}

impl Batch {
    /// Get a tensor by name.
    pub fn get(&self, name: &str) -> Option<TensorNodeId> {
        self.tensors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| *t)
    }

    /// Get the "input" tensor.
    pub fn input(&self) -> Option<TensorNodeId> {
        self.get("input")
    }

    /// Get the "target" tensor.
    pub fn target(&self) -> Option<TensorNodeId> {
        self.get("target")
    }
}

// ── Samplers ─────────────────────────────────────────────────────────────

/// A simple LCG-based RNG for deterministic shuffling without external deps.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    /// Generate a random usize in [0, bound).
    fn next_below(&mut self, bound: usize) -> usize {
        (self.next_u64() >> 33) as usize % bound
    }

    /// Fisher-Yates shuffle on a mutable slice.
    fn shuffle(&mut self, slice: &mut [usize]) {
        let n = slice.len();
        for i in (1..n).rev() {
            let j = self.next_below(i + 1);
            slice.swap(i, j);
        }
    }
}

fn refill_sequential_indices(indices: &mut [usize]) {
    for (index, value) in indices.iter_mut().enumerate() {
        *value = index;
    }
}

#[inline]
fn threshold_index_4(
    u: f64,
    first_threshold: f64,
    second_threshold: f64,
    third_threshold: f64,
) -> usize {
    usize::from(u > first_threshold)
        + usize::from(u > second_threshold)
        + usize::from(u > third_threshold)
}

#[inline]
fn finite_threshold_cmp(threshold: f64, sample: f64) -> std::cmp::Ordering {
    if threshold < sample {
        std::cmp::Ordering::Less
    } else if threshold > sample {
        std::cmp::Ordering::Greater
    } else {
        std::cmp::Ordering::Equal
    }
}

fn threshold_binary_index(cumulative: &[f64], sample: f64) -> usize {
    match cumulative.binary_search_by(|c| finite_threshold_cmp(*c, sample)) {
        Ok(i) => i,
        Err(i) => i.min(cumulative.len() - 1),
    }
}

struct ThresholdBucketIndex {
    starts: Vec<usize>,
}

impl ThresholdBucketIndex {
    const MIN_THRESHOLDS: usize = 64;
    const MAX_BUCKETS: usize = 4096;
    const MAX_BUCKET_SPAN: usize = 64;

    fn new(cumulative: &[f64]) -> Option<Self> {
        if cumulative.len() < Self::MIN_THRESHOLDS || cumulative.windows(2).any(|w| w[0] >= w[1]) {
            return None;
        }

        let bucket_count = cumulative.len().min(Self::MAX_BUCKETS);
        let inv_bucket_count = 1.0 / bucket_count as f64;
        let mut starts = Vec::with_capacity(bucket_count);
        let mut cursor = 0usize;
        for bucket in 0..bucket_count {
            let bucket_start = bucket as f64 * inv_bucket_count;
            while cursor < cumulative.len() && cumulative[cursor] < bucket_start {
                cursor += 1;
            }
            starts.push(cursor.min(cumulative.len() - 1));
        }

        for bucket in 0..bucket_count {
            let start = starts[bucket];
            let end = if bucket + 1 < bucket_count {
                starts[bucket + 1]
            } else {
                cumulative.len() - 1
            };
            if end.saturating_sub(start) + 1 > Self::MAX_BUCKET_SPAN {
                return None;
            }
        }

        Some(Self { starts })
    }

    fn find(&self, cumulative: &[f64], sample: f64) -> usize {
        let mut bucket = (sample * self.starts.len() as f64) as usize;
        if bucket >= self.starts.len() {
            bucket = self.starts.len() - 1;
        }
        let mut index = self.starts[bucket];
        while index + 1 < cumulative.len() && cumulative[index] < sample {
            index += 1;
        }
        index
    }
}

/// Yields sample indices sequentially: 0, 1, 2, ..., n-1.
pub struct SequentialSampler {
    size: usize,
}

impl SequentialSampler {
    pub fn new(size: usize) -> Self {
        Self { size }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn indices(&self) -> Vec<usize> {
        (0..self.size).collect()
    }
}

/// Yields a random permutation of 0..n, or samples with replacement.
pub struct RandomSampler {
    size: usize,
    replacement: bool,
    num_samples: usize,
    seed: u64,
}

impl RandomSampler {
    pub fn new(size: usize) -> Self {
        Self {
            size,
            replacement: false,
            num_samples: size,
            seed: 0xCAFE_BABE_1234_5678,
        }
    }

    pub fn with_replacement(mut self, replacement: bool) -> Self {
        self.replacement = replacement;
        self
    }

    pub fn with_num_samples(mut self, num_samples: usize) -> Self {
        self.num_samples = num_samples;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn len(&self) -> usize {
        if self.size == 0 {
            return 0;
        }
        self.num_samples
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn indices(&self) -> Vec<usize> {
        if self.size == 0 {
            return Vec::new();
        }
        let mut rng = SimpleRng::new(self.seed);
        if self.replacement {
            (0..self.num_samples)
                .map(|_| rng.next_below(self.size))
                .collect()
        } else {
            let mut result = Vec::with_capacity(self.num_samples);
            let mut idx: Vec<usize> = (0..self.size).collect();
            let full_passes = self.num_samples / self.size;
            let remainder = self.num_samples % self.size;
            let reset_count =
                full_passes.saturating_sub(1) + usize::from(remainder > 0 && full_passes > 0);
            let base_indices = (reset_count >= 2).then(|| idx.clone());
            for pass in 0..full_passes {
                if pass > 0 {
                    if let Some(base) = base_indices.as_deref() {
                        idx.copy_from_slice(base);
                    } else {
                        refill_sequential_indices(&mut idx);
                    }
                }
                rng.shuffle(&mut idx);
                result.extend_from_slice(&idx);
            }
            if remainder > 0 {
                if full_passes > 0 {
                    if let Some(base) = base_indices.as_deref() {
                        idx.copy_from_slice(base);
                    } else {
                        refill_sequential_indices(&mut idx);
                    }
                }
                rng.shuffle(&mut idx);
                result.extend_from_slice(&idx[..remainder]);
            }
            result
        }
    }
}

/// Yields a random permutation of a given subset of indices.
pub struct SubsetRandomSampler {
    indices: Vec<usize>,
    seed: u64,
}

impl SubsetRandomSampler {
    pub fn new(indices: Vec<usize>) -> Self {
        Self {
            indices,
            seed: 0xBEEF_DEAD_0000_1111,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn len(&self) -> usize {
        self.indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    pub fn indices(&self) -> Vec<usize> {
        let mut result = self.indices.clone();
        let mut rng = SimpleRng::new(self.seed);
        rng.shuffle(&mut result);
        result
    }
}

/// Samples indices with probability proportional to given weights.
///
/// Uses replacement by default (can sample same index multiple times).
pub struct WeightedRandomSampler {
    weights: Vec<f64>,
    num_samples: usize,
    seed: u64,
}

impl WeightedRandomSampler {
    pub fn new(weights: Vec<f64>, num_samples: usize) -> Self {
        Self {
            weights,
            num_samples,
            seed: 0xFEED_FACE_DEAD_BEEF,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn len(&self) -> usize {
        self.num_samples
    }

    pub fn is_empty(&self) -> bool {
        self.num_samples == 0
    }

    pub fn indices(&self) -> Result<Vec<usize>, AutogradError> {
        if self.num_samples == 0 {
            return Err(transform_config_error(
                "WeightedRandomSampler: num_samples must be > 0",
            ));
        }
        if self.weights.is_empty() {
            return Err(transform_config_error(
                "WeightedRandomSampler: weights must be non-empty",
            ));
        }
        let mut cumulative = Vec::with_capacity(self.weights.len());
        let mut total = 0.0;
        for &w in &self.weights {
            if !w.is_finite() {
                return Err(transform_config_error(
                    "WeightedRandomSampler: weights must be finite",
                ));
            }
            if w < 0.0 {
                return Err(transform_config_error(
                    "WeightedRandomSampler: weights must be non-negative",
                ));
            }
            total += w;
            if !total.is_finite() {
                return Err(transform_config_error(
                    "WeightedRandomSampler: sum of weights must be finite",
                ));
            }
            cumulative.push(total);
        }

        if total <= 0.0 {
            return Err(transform_config_error(
                "WeightedRandomSampler: sum of weights must be > 0",
            ));
        }

        if self.weights.len() == 1 {
            return Ok(vec![0; self.num_samples]);
        }

        if self.weights.len() == 2 {
            let first_threshold = cumulative.first().copied().ok_or_else(|| {
                transform_config_error(
                    "WeightedRandomSampler: weights must produce cumulative thresholds",
                )
            })? / total;
            let mut rng = SimpleRng::new(self.seed);
            let mut result = Vec::with_capacity(self.num_samples);
            for _ in 0..self.num_samples {
                let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                result.push(if u <= first_threshold { 0 } else { 1 });
            }
            return Ok(result);
        }

        if self.weights.len() == 3 {
            let first_threshold = cumulative.first().copied().ok_or_else(|| {
                transform_config_error(
                    "WeightedRandomSampler: weights must produce cumulative thresholds",
                )
            })? / total;
            let second_threshold = cumulative.get(1).copied().ok_or_else(|| {
                transform_config_error(
                    "WeightedRandomSampler: weights must produce cumulative thresholds",
                )
            })? / total;
            let mut rng = SimpleRng::new(self.seed);
            let mut result = Vec::with_capacity(self.num_samples);
            for _ in 0..self.num_samples {
                let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                result.push(if u <= first_threshold {
                    0
                } else if u <= second_threshold {
                    1
                } else {
                    2
                });
            }
            return Ok(result);
        }

        if self.weights.len() == 4 {
            let first_threshold = cumulative.first().copied().ok_or_else(|| {
                transform_config_error(
                    "WeightedRandomSampler: weights must produce cumulative thresholds",
                )
            })? / total;
            let second_threshold = cumulative.get(1).copied().ok_or_else(|| {
                transform_config_error(
                    "WeightedRandomSampler: weights must produce cumulative thresholds",
                )
            })? / total;
            let third_threshold = cumulative.get(2).copied().ok_or_else(|| {
                transform_config_error(
                    "WeightedRandomSampler: weights must produce cumulative thresholds",
                )
            })? / total;
            let mut rng = SimpleRng::new(self.seed);
            let mut result = Vec::with_capacity(self.num_samples);
            for _ in 0..self.num_samples {
                let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                result.push(threshold_index_4(
                    u,
                    first_threshold,
                    second_threshold,
                    third_threshold,
                ));
            }
            return Ok(result);
        }

        for threshold in &mut cumulative {
            *threshold /= total;
        }

        let mut rng = SimpleRng::new(self.seed);
        let mut result = Vec::with_capacity(self.num_samples);
        let bucket_index = ThresholdBucketIndex::new(&cumulative);
        for _ in 0..self.num_samples {
            let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let idx = if let Some(index) = &bucket_index {
                index.find(&cumulative, u)
            } else {
                threshold_binary_index(&cumulative, u)
            };
            result.push(idx);
        }
        Ok(result)
    }
}

/// Groups sampler indices into batches of a given size.
pub struct BatchSampler {
    indices: Vec<usize>,
    batch_size: usize,
    drop_last: bool,
}

impl BatchSampler {
    /// Create a `BatchSampler` from a pre-computed list of indices.
    pub fn new(indices: Vec<usize>, batch_size: usize, drop_last: bool) -> Self {
        Self {
            indices,
            batch_size,
            drop_last,
        }
    }

    /// Create from a `SequentialSampler`.
    pub fn from_sequential(
        sampler: &SequentialSampler,
        batch_size: usize,
        drop_last: bool,
    ) -> Self {
        Self::new(sampler.indices(), batch_size, drop_last)
    }

    /// Create from a `RandomSampler`.
    pub fn from_random(sampler: &RandomSampler, batch_size: usize, drop_last: bool) -> Self {
        Self::new(sampler.indices(), batch_size, drop_last)
    }

    /// Create from a `SubsetRandomSampler`.
    pub fn from_subset(sampler: &SubsetRandomSampler, batch_size: usize, drop_last: bool) -> Self {
        Self::new(sampler.indices(), batch_size, drop_last)
    }

    /// Number of batches.
    pub fn len(&self) -> usize {
        if self.indices.is_empty() || self.batch_size == 0 {
            return 0;
        }
        if self.drop_last {
            self.indices.len() / self.batch_size
        } else {
            self.indices.len().div_ceil(self.batch_size)
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return all batches as vectors of indices.
    pub fn batches(&self) -> Vec<Vec<usize>> {
        if self.batch_size == 0 {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(self.len());
        let mut pos = 0;
        while pos < self.indices.len() {
            let end = (pos + self.batch_size).min(self.indices.len());
            let batch = self.indices[pos..end].to_vec();
            if batch.len() < self.batch_size && self.drop_last {
                break;
            }
            result.push(batch);
            pos = end;
        }
        result
    }
}

// ── DataLoader ───────────────────────────────────────────────────────────

/// Configuration for a `DataLoader`.
pub struct DataLoaderConfig {
    /// Number of samples per batch.
    pub batch_size: usize,
    /// Whether to shuffle data each epoch.
    pub shuffle: bool,
    /// Whether to drop the last incomplete batch.
    pub drop_last: bool,
}

impl DataLoaderConfig {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            shuffle: false,
            drop_last: false,
        }
    }

    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn with_drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }
}

/// Iterates over a dataset in batches, optionally shuffling.
///
/// Each call to `next_batch()` returns a `Batch` of collated tensors
/// registered in the provided `FrankenTorchSession`.
pub struct DataLoader<'a, D: Dataset> {
    dataset: &'a D,
    config: DataLoaderConfig,
    indices: Vec<usize>,
    position: usize,
    auto_shuffle: bool,
    pending_shuffle: bool,
    /// Simple LCG state for shuffling (deterministic, no external deps).
    rng_state: u64,
}

impl<'a, D: Dataset> DataLoader<'a, D> {
    /// Create a new `DataLoader` for the given dataset and config.
    pub fn new(dataset: &'a D, config: DataLoaderConfig) -> Self {
        let n = dataset.len();
        let indices: Vec<usize> = (0..n).collect();
        let auto_shuffle = config.shuffle;
        Self {
            dataset,
            config,
            indices,
            position: 0,
            auto_shuffle,
            pending_shuffle: auto_shuffle,
            rng_state: 0xDEAD_BEEF_CAFE_1234,
        }
    }

    /// Create a DataLoader that uses a custom sampler's index ordering.
    pub fn with_indices(dataset: &'a D, indices: Vec<usize>, config: DataLoaderConfig) -> Self {
        Self {
            dataset,
            config,
            indices,
            position: 0,
            auto_shuffle: false,
            pending_shuffle: false,
            rng_state: 0xDEAD_BEEF_CAFE_1234,
        }
    }

    /// Set the random seed for shuffling.
    pub fn seed(mut self, seed: u64) -> Self {
        self.rng_state = seed;
        if self.auto_shuffle && self.position == 0 {
            self.pending_shuffle = true;
        }
        self
    }

    /// Reset the loader to the beginning of the dataset.
    /// If shuffle is enabled, re-shuffles the indices.
    pub fn reset(&mut self) {
        self.position = 0;
        if self.auto_shuffle {
            self.pending_shuffle = true;
        }
    }

    /// Number of batches in one epoch.
    pub fn num_batches(&self) -> usize {
        let n = self.indices.len();
        if n == 0 || self.config.batch_size == 0 {
            return 0;
        }
        if self.config.drop_last {
            n / self.config.batch_size
        } else {
            n.div_ceil(self.config.batch_size)
        }
    }

    /// Get the next batch, collating samples into tensors.
    ///
    /// Returns `None` when all batches have been consumed for this epoch.
    /// Call `reset()` to start a new epoch.
    pub fn next_batch(
        &mut self,
        session: &mut FrankenTorchSession,
    ) -> Result<Option<Batch>, AutogradError> {
        if self.pending_shuffle {
            self.shuffle_indices();
            self.pending_shuffle = false;
        }
        let n = self.indices.len();
        if self.position >= n || self.config.batch_size == 0 {
            return Ok(None);
        }

        let remaining = n - self.position;
        let batch_size = remaining.min(self.config.batch_size);

        // Drop incomplete final batch if configured
        if batch_size < self.config.batch_size && self.config.drop_last {
            self.position = n;
            return Ok(None);
        }

        let dataset_len = self.dataset.len();
        let batch_start = self.position;
        let batch_end = batch_start + batch_size;
        let batch_indices = &self.indices[batch_start..batch_end];
        for &idx in batch_indices {
            if idx >= dataset_len {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "DataLoader: sampler index out of range for dataset",
                    },
                )));
            }
        }
        self.position = batch_end;

        if let Some(batch) = self.dataset.collate_indices(batch_indices, session) {
            return batch.map(Some);
        }

        // Collect samples for this batch
        let mut samples = Vec::with_capacity(batch_size);
        for &idx in batch_indices {
            samples.push(self.dataset.get(idx));
        }

        // Collate: stack tensors along a new batch dimension
        let batch = collate(session, &samples, batch_size)?;
        Ok(Some(batch))
    }

    fn shuffle_indices(&mut self) {
        // Fisher-Yates shuffle with simple LCG
        let n = self.indices.len();
        for i in (1..n).rev() {
            self.rng_state = self
                .rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let j = (self.rng_state >> 33) as usize % (i + 1);
            self.indices.swap(i, j);
        }
    }
}

/// Collate a batch of `DataItem`s into a single `Batch` with stacked tensors.
///
/// All items must have the same number of tensors with the same names and shapes.
fn collate(
    session: &mut FrankenTorchSession,
    samples: &[DataItem],
    batch_size: usize,
) -> Result<Batch, AutogradError> {
    if samples.is_empty() {
        return Ok(Batch {
            tensors: Vec::new(),
        });
    }

    let num_tensors = samples[0].tensors.len();
    let mut batch_tensors = Vec::with_capacity(num_tensors);

    for tensor_idx in 0..num_tensors {
        let (ref name, _, ref first_shape) = samples[0].tensors[tensor_idx];

        // Validate all samples have matching shapes for this tensor
        let sample_numel =
            checked_shape_numel(first_shape, "DataLoader: sample shape volume overflow")?;
        let first_values_len = samples[0].tensors[tensor_idx].1.len();
        if first_values_len != sample_numel {
            return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet {
                    reason: "DataLoader: sample values length does not match declared tensor shape",
                },
            )));
        }
        for sample in samples.iter().skip(1) {
            if sample.tensors.len() != num_tensors {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "DataLoader: samples have different numbers of tensors",
                    },
                )));
            }
            let (ref s_name, ref s_vals, ref s_shape) = sample.tensors[tensor_idx];
            if s_name != name {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "DataLoader: tensor names differ across samples in batch",
                    },
                )));
            }
            if s_shape != first_shape {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "DataLoader: tensor shapes differ across samples in batch",
                    },
                )));
            }
            if s_vals.len() != sample_numel {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "DataLoader: sample values length does not match declared tensor shape",
                    },
                )));
            }
        }

        // Stack along batch dimension: new shape = [batch_size] ++ sample_shape
        let batch_numel = checked_mul(batch_size, sample_numel, "DataLoader: batch size overflow")?;
        let mut batched_values = Vec::with_capacity(batch_numel);
        for sample in samples {
            let (_, ref vals, _) = sample.tensors[tensor_idx];
            batched_values.extend_from_slice(vals);
        }

        let mut batched_shape = Vec::with_capacity(1 + first_shape.len());
        batched_shape.push(batch_size);
        batched_shape.extend_from_slice(first_shape);

        let tensor = session.tensor_variable(batched_values, batched_shape, false)?;
        batch_tensors.push((name.clone(), tensor));
    }

    Ok(Batch {
        tensors: batch_tensors,
    })
}

fn collate_tensor_dataset_items(
    session: &mut FrankenTorchSession,
    items: &[DataItem],
    indices: &[usize],
) -> Result<Batch, AutogradError> {
    if indices.is_empty() {
        return Ok(Batch {
            tensors: Vec::new(),
        });
    }

    let first = items
        .get(indices[0])
        .ok_or_else(|| dataloader_error("DataLoader: sampler index out of range for dataset"))?;
    let num_tensors = first.tensors.len();
    let mut batch_tensors = Vec::with_capacity(num_tensors);

    for tensor_idx in 0..num_tensors {
        let (ref name, ref first_values, ref first_shape) = first.tensors[tensor_idx];
        let sample_numel =
            checked_shape_numel(first_shape, "DataLoader: sample shape volume overflow")?;
        if first_values.len() != sample_numel {
            return Err(dataloader_error(
                "DataLoader: sample values length does not match declared tensor shape",
            ));
        }

        for &sample_index in indices.iter().skip(1) {
            let sample = items.get(sample_index).ok_or_else(|| {
                dataloader_error("DataLoader: sampler index out of range for dataset")
            })?;
            if sample.tensors.len() != num_tensors {
                return Err(dataloader_error(
                    "DataLoader: samples have different numbers of tensors",
                ));
            }
            let (ref sample_name, ref sample_values, ref sample_shape) = sample.tensors[tensor_idx];
            if sample_name != name {
                return Err(dataloader_error(
                    "DataLoader: tensor names differ across samples in batch",
                ));
            }
            if sample_shape != first_shape {
                return Err(dataloader_error(
                    "DataLoader: tensor shapes differ across samples in batch",
                ));
            }
            if sample_values.len() != sample_numel {
                return Err(dataloader_error(
                    "DataLoader: sample values length does not match declared tensor shape",
                ));
            }
        }

        let batch_size = indices.len();
        let batch_numel = checked_mul(batch_size, sample_numel, "DataLoader: batch size overflow")?;
        let mut batched_values = Vec::with_capacity(batch_numel);
        for &sample_index in indices {
            let sample = items.get(sample_index).ok_or_else(|| {
                dataloader_error("DataLoader: sampler index out of range for dataset")
            })?;
            let (_, ref values, _) = sample.tensors[tensor_idx];
            batched_values.extend_from_slice(values);
        }

        let mut batched_shape = Vec::with_capacity(1 + first_shape.len());
        batched_shape.push(batch_size);
        batched_shape.extend_from_slice(first_shape);

        let tensor = session.tensor_variable(batched_values, batched_shape, false)?;
        batch_tensors.push((name.clone(), tensor));
    }

    Ok(Batch {
        tensors: batch_tensors,
    })
}

fn contiguous_index_start(indices: &[usize]) -> Option<usize> {
    let (&first, rest) = indices.split_first()?;
    for (offset, &index) in rest.iter().enumerate() {
        let expected = offset
            .checked_add(1)
            .and_then(|step| first.checked_add(step))?;
        if index != expected {
            return None;
        }
    }
    Some(first)
}

fn collate_tensor_dataset_columns(
    session: &mut FrankenTorchSession,
    columns: &[TensorDatasetColumn],
    dataset_len: usize,
    indices: &[usize],
) -> Result<Batch, AutogradError> {
    if indices.is_empty() {
        return Ok(Batch {
            tensors: Vec::new(),
        });
    }

    for &sample_index in indices {
        if sample_index >= dataset_len {
            return Err(dataloader_error(
                "DataLoader: sampler index out of range for dataset",
            ));
        }
    }

    let batch_size = indices.len();
    let contiguous_start = contiguous_index_start(indices);
    let mut batch_tensors = Vec::with_capacity(columns.len());

    for column in columns {
        let batch_numel = checked_mul(
            batch_size,
            column.sample_numel,
            "DataLoader: batch size overflow",
        )?;
        let batched_values = if let Some(start) = contiguous_start {
            let start_offset = checked_mul(
                start,
                column.sample_numel,
                "DataLoader: batch size overflow",
            )?;
            let end_offset = start_offset
                .checked_add(batch_numel)
                .ok_or_else(|| dataloader_error("DataLoader: batch size overflow"))?;
            column.values[start_offset..end_offset].to_vec()
        } else {
            let mut values = Vec::with_capacity(batch_numel);
            for &sample_index in indices {
                let start_offset = checked_mul(
                    sample_index,
                    column.sample_numel,
                    "DataLoader: batch size overflow",
                )?;
                let end_offset = start_offset
                    .checked_add(column.sample_numel)
                    .ok_or_else(|| dataloader_error("DataLoader: batch size overflow"))?;
                values.extend_from_slice(&column.values[start_offset..end_offset]);
            }
            values
        };

        let mut batched_shape = Vec::with_capacity(1 + column.shape.len());
        batched_shape.push(batch_size);
        batched_shape.extend_from_slice(&column.shape);

        let tensor = session.tensor_variable(batched_values, batched_shape, false)?;
        batch_tensors.push((column.name.clone(), tensor));
    }

    Ok(Batch {
        tensors: batch_tensors,
    })
}

// ── Transforms ──────────────────────────────────────────────────────────

/// A transform that can be applied to a `DataItem`.
///
/// Equivalent to `torchvision.transforms` applied to tensors.
pub trait Transform {
    /// Apply the transform to a data item, returning a modified copy.
    fn apply(&self, item: DataItem) -> DataItem;
}

/// Compose multiple transforms into a pipeline.
///
/// Equivalent to `torchvision.transforms.Compose`.
/// Applies transforms in order.
pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }
}

impl Transform for Compose {
    fn apply(&self, mut item: DataItem) -> DataItem {
        for t in &self.transforms {
            item = t.apply(item);
        }
        item
    }
}

/// Normalize a named tensor in a data item by subtracting mean and dividing by std.
///
/// Equivalent to `torchvision.transforms.Normalize(mean, std)`.
/// Applied per-channel: `output[c] = (input[c] - mean[c]) / std[c]`.
#[derive(Debug)]
pub struct NormalizeTransform {
    /// Name of the tensor to normalize (e.g., "input").
    tensor_name: String,
    /// Per-channel means.
    mean: Vec<f64>,
    /// Per-channel standard deviations.
    std: Vec<f64>,
}

impl NormalizeTransform {
    pub fn new(tensor_name: &str, mean: Vec<f64>, std: Vec<f64>) -> Result<Self, AutogradError> {
        if mean.len() != std.len() {
            return Err(transform_config_error(
                "NormalizeTransform requires mean/std lengths to match",
            ));
        }
        if mean.iter().any(|value| !value.is_finite()) {
            return Err(transform_config_error(
                "NormalizeTransform requires finite mean values",
            ));
        }
        if std.iter().any(|value| !value.is_finite() || *value <= 0.0) {
            return Err(transform_config_error(
                "NormalizeTransform requires finite std values > 0",
            ));
        }
        Ok(Self {
            tensor_name: tensor_name.to_string(),
            mean,
            std,
        })
    }
}

impl Transform for NormalizeTransform {
    fn apply(&self, mut item: DataItem) -> DataItem {
        for (name, values, shape) in &mut item.tensors {
            if name == &self.tensor_name && !self.mean.is_empty() {
                let channels = self.mean.len();
                let channel_count_matches = shape
                    .first()
                    .copied()
                    .is_some_and(|first| first.cmp(&channels).is_eq());
                if !channel_count_matches || values.len() % channels > 0 {
                    continue;
                }
                let channel_size = values.len() / channels;
                for c in 0..channels {
                    let m = self.mean[c];
                    let s = self.std[c];
                    for i in 0..channel_size {
                        let idx = c * channel_size + i;
                        if idx < values.len() {
                            values[idx] = (values[idx] - m) / s;
                        }
                    }
                }
            }
        }
        item
    }
}

/// Apply an arbitrary function to a data item.
///
/// Equivalent to `torchvision.transforms.Lambda`.
pub struct LambdaTransform {
    func: Box<dyn Fn(DataItem) -> DataItem>,
}

impl LambdaTransform {
    pub fn new(func: impl Fn(DataItem) -> DataItem + 'static) -> Self {
        Self {
            func: Box::new(func),
        }
    }
}

impl Transform for LambdaTransform {
    fn apply(&self, item: DataItem) -> DataItem {
        (self.func)(item)
    }
}

/// Scale all values in a named tensor by a constant factor.
pub struct ScaleTransform {
    tensor_name: String,
    factor: f64,
}

impl ScaleTransform {
    pub fn new(tensor_name: &str, factor: f64) -> Self {
        Self {
            tensor_name: tensor_name.to_string(),
            factor,
        }
    }
}

impl Transform for ScaleTransform {
    fn apply(&self, mut item: DataItem) -> DataItem {
        for (name, values, _shape) in &mut item.tensors {
            if name == &self.tensor_name {
                for v in values.iter_mut() {
                    *v *= self.factor;
                }
            }
        }
        item
    }
}

/// A dataset wrapper that applies a transform to each item.
///
/// Equivalent to applying `transform` in a PyTorch `Dataset.__getitem__`.
pub struct TransformDataset<D: Dataset> {
    inner: D,
    transform: Box<dyn Transform>,
}

impl<D: Dataset> TransformDataset<D> {
    pub fn new(dataset: D, transform: Box<dyn Transform>) -> Self {
        Self {
            inner: dataset,
            transform,
        }
    }
}

impl<D: Dataset> Dataset for TransformDataset<D> {
    fn len(&self) -> usize {
        self.inner.len()
    }

    fn get(&self, index: usize) -> DataItem {
        let item = self.inner.get(index);
        self.transform.apply(item)
    }
}

// ── Subset and random_split ─────────────────────────────────────────────

/// A dataset wrapping a subset of another dataset by index.
///
/// Equivalent to `torch.utils.data.Subset`.
pub struct Subset<D: Dataset> {
    dataset: std::sync::Arc<D>,
    indices: Vec<usize>,
}

impl<D: Dataset> Subset<D> {
    /// Create a subset from an Arc-wrapped dataset and indices.
    pub fn new(dataset: std::sync::Arc<D>, indices: Vec<usize>) -> Self {
        Self { dataset, indices }
    }

    /// Return the indices used by this subset.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }
}

impl<D: Dataset> Dataset for Subset<D> {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> DataItem {
        self.dataset.get(self.indices[index])
    }
}

/// Split a dataset into non-overlapping subsets of given lengths.
///
/// Equivalent to `torch.utils.data.random_split(dataset, lengths)`.
///
/// Uses a deterministic Fisher-Yates shuffle seeded by `seed`.
/// The underlying dataset is shared via `Arc` (no data cloning).
pub fn random_split<D: Dataset>(dataset: D, lengths: &[usize], seed: u64) -> Vec<Subset<D>> {
    let n = dataset.len();
    let total: usize = lengths.iter().sum();
    assert_eq!(
        total, n,
        "random_split: sum of lengths ({total}) must equal dataset size ({n})",
    );

    let shared = std::sync::Arc::new(dataset);

    // Generate random permutation with LCG
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng_state = seed;
    for i in (1..indices.len()).rev() {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let j = (rng_state >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }

    let mut subsets = Vec::with_capacity(lengths.len());
    let mut offset = 0;
    for &len in lengths {
        subsets.push(Subset::new(
            std::sync::Arc::clone(&shared),
            indices[offset..offset + len].to_vec(),
        ));
        offset += len;
    }
    subsets
}

#[cfg(test)]
mod tests {
    use ft_core::ExecutionMode;

    use super::*;

    fn make_dataset(n: usize, features: usize) -> TensorDataset {
        let items: Vec<DataItem> = (0..n)
            .map(|i| {
                let input: Vec<f64> = (0..features).map(|f| (i * features + f) as f64).collect();
                let target = vec![i as f64];
                DataItem::input_target(input, vec![features], target, vec![1])
            })
            .collect();
        TensorDataset::new(items)
    }

    #[test]
    fn dataset_len() {
        let ds = make_dataset(10, 3);
        assert_eq!(ds.len(), 10);
        assert!(!ds.is_empty());
    }

    #[test]
    fn dataset_get() {
        let ds = make_dataset(5, 2);
        let item = ds.get(2);
        assert_eq!(item.tensors.len(), 2);
        assert_eq!(item.tensors[0].0, "input");
        assert_eq!(item.tensors[0].1, vec![4.0, 5.0]); // i=2, features=2: 4,5
        assert_eq!(item.tensors[1].0, "target");
        assert_eq!(item.tensors[1].1, vec![2.0]);
    }

    #[test]
    fn dataset_from_inputs_targets() {
        let inputs = vec![(vec![1.0, 2.0], vec![2]), (vec![3.0, 4.0], vec![2])];
        let targets = vec![(vec![0.0], vec![1]), (vec![1.0], vec![1])];
        let ds = TensorDataset::from_inputs_targets(inputs, targets);
        assert_eq!(ds.len(), 2);
        let item = ds.get(1);
        assert_eq!(item.tensors[0].1, vec![3.0, 4.0]);
        assert_eq!(item.tensors[1].1, vec![1.0]);
    }

    #[test]
    fn dataloader_num_batches() {
        let ds = make_dataset(10, 3);
        let config = DataLoaderConfig::new(3);
        let loader = DataLoader::new(&ds, config);
        // ceil(10/3) = 4 batches (3+3+3+1)
        assert_eq!(loader.num_batches(), 4);
    }

    #[test]
    fn dataloader_num_batches_drop_last() {
        let ds = make_dataset(10, 3);
        let config = DataLoaderConfig::new(3).with_drop_last(true);
        let loader = DataLoader::new(&ds, config);
        // floor(10/3) = 3 batches (drop the 1-element last batch)
        assert_eq!(loader.num_batches(), 3);
    }

    #[test]
    fn dataloader_iterates_all_samples() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(7, 2);
        let config = DataLoaderConfig::new(3);
        let mut loader = DataLoader::new(&ds, config);

        let mut total_samples = 0;
        let mut batches = 0;
        while let Some(batch) = loader.next_batch(&mut session).unwrap() {
            let input = batch.input().unwrap();
            let shape = session.tensor_shape(input).unwrap();
            total_samples += shape[0];
            batches += 1;
        }
        assert_eq!(batches, 3); // 3+3+1
        assert_eq!(total_samples, 7);
    }

    #[test]
    fn dataloader_batch_shape() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(6, 4);
        let config = DataLoaderConfig::new(3);
        let mut loader = DataLoader::new(&ds, config);

        let batch = loader.next_batch(&mut session).unwrap().unwrap();
        let input = batch.input().unwrap();
        let target = batch.target().unwrap();
        assert_eq!(session.tensor_shape(input).unwrap(), vec![3, 4]);
        assert_eq!(session.tensor_shape(target).unwrap(), vec![3, 1]);
    }

    #[test]
    fn dataloader_batch_values_correct() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(4, 2);
        let config = DataLoaderConfig::new(2);
        let mut loader = DataLoader::new(&ds, config);

        // First batch: samples 0 and 1
        let batch = loader.next_batch(&mut session).unwrap().unwrap();
        let input = batch.input().unwrap();
        let vals = session.tensor_values(input).unwrap();
        // sample 0: [0, 1], sample 1: [2, 3]
        assert_eq!(vals, vec![0.0, 1.0, 2.0, 3.0]);

        // Second batch: samples 2 and 3
        let batch2 = loader.next_batch(&mut session).unwrap().unwrap();
        let input2 = batch2.input().unwrap();
        let vals2 = session.tensor_values(input2).unwrap();
        assert_eq!(vals2, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn tensor_dataset_columnar_cache_preserves_contiguous_batch_values() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(4, 2);
        assert!(ds.columns.is_some());
        let mut loader = DataLoader::new(&ds, DataLoaderConfig::new(2));

        let batch = loader.next_batch(&mut session).unwrap().unwrap();
        let input = batch.input().unwrap();
        let target = batch.target().unwrap();

        assert_eq!(
            session.tensor_values(input).unwrap(),
            vec![0.0, 1.0, 2.0, 3.0]
        );
        assert_eq!(session.tensor_values(target).unwrap(), vec![0.0, 1.0]);
        assert_eq!(session.tensor_shape(input).unwrap(), vec![2, 2]);
        assert_eq!(session.tensor_shape(target).unwrap(), vec![2, 1]);
    }

    #[test]
    fn tensor_dataset_columnar_cache_preserves_custom_index_order() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(4, 2);
        assert!(ds.columns.is_some());
        let mut loader = DataLoader::with_indices(&ds, vec![2, 0], DataLoaderConfig::new(2));

        let batch = loader.next_batch(&mut session).unwrap().unwrap();
        let input = batch.input().unwrap();
        let target = batch.target().unwrap();

        assert_eq!(
            session.tensor_values(input).unwrap(),
            vec![4.0, 5.0, 0.0, 1.0]
        );
        assert_eq!(session.tensor_values(target).unwrap(), vec![2.0, 0.0]);
    }

    #[test]
    fn tensor_dataset_columnar_cache_falls_back_for_mismatched_shapes() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = TensorDataset::new(vec![
            DataItem::single("input", vec![1.0, 2.0], vec![2]),
            DataItem::single("input", vec![3.0, 4.0], vec![1, 2]),
        ]);
        assert!(ds.columns.is_none());
        let mut loader = DataLoader::new(&ds, DataLoaderConfig::new(2));

        let message = loader
            .next_batch(&mut session)
            .err()
            .map(|err| err.to_string())
            .unwrap_or_default();
        assert!(
            message.contains("tensor shapes differ"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn dataloader_drop_last() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(5, 2);
        let config = DataLoaderConfig::new(3).with_drop_last(true);
        let mut loader = DataLoader::new(&ds, config);

        // Should only get 1 batch of 3, dropping the remaining 2
        let batch = loader.next_batch(&mut session).unwrap().unwrap();
        let shape = session.tensor_shape(batch.input().unwrap()).unwrap();
        assert_eq!(shape[0], 3);

        // No more batches
        assert!(loader.next_batch(&mut session).unwrap().is_none());
    }

    #[test]
    fn dataloader_shuffle_produces_different_order() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(20, 1);
        let config1 = DataLoaderConfig::new(20).with_shuffle(true);
        let mut loader1 = DataLoader::new(&ds, config1).seed(42);
        loader1.reset(); // triggers shuffle

        let batch1 = loader1.next_batch(&mut session).unwrap().unwrap();
        let vals1 = session.tensor_values(batch1.input().unwrap()).unwrap();

        // Second epoch: should be different order
        loader1.reset();
        let batch2 = loader1.next_batch(&mut session).unwrap().unwrap();
        let vals2 = session.tensor_values(batch2.input().unwrap()).unwrap();

        // With 20 elements, the probability of same order after shuffle is ~1/20!
        assert_ne!(
            vals1, vals2,
            "shuffled epochs should produce different orders"
        );
    }

    #[test]
    fn dataloader_shuffle_applies_on_first_epoch_without_reset() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(8, 1);
        let config = DataLoaderConfig::new(8).with_shuffle(true);

        let mut loader_a = DataLoader::new(&ds, config).seed(42);
        let mut loader_b =
            DataLoader::new(&ds, DataLoaderConfig::new(8).with_shuffle(true)).seed(42);

        let batch_a = loader_a.next_batch(&mut session).unwrap().unwrap();
        let vals_a = session.tensor_values(batch_a.input().unwrap()).unwrap();
        let batch_b = loader_b.next_batch(&mut session).unwrap().unwrap();
        let vals_b = session.tensor_values(batch_b.input().unwrap()).unwrap();

        assert_ne!(vals_a, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(
            vals_a, vals_b,
            "same seed should control first-epoch shuffle"
        );
    }

    #[test]
    fn dataloader_no_shuffle_same_order() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(5, 1);

        let config = DataLoaderConfig::new(5);
        let mut loader = DataLoader::new(&ds, config);

        let batch1 = loader.next_batch(&mut session).unwrap().unwrap();
        let vals1 = session.tensor_values(batch1.input().unwrap()).unwrap();

        loader.reset();
        let batch2 = loader.next_batch(&mut session).unwrap().unwrap();
        let vals2 = session.tensor_values(batch2.input().unwrap()).unwrap();

        assert_eq!(vals1, vals2);
    }

    #[test]
    fn dataloader_empty_dataset() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = TensorDataset::new(Vec::new());
        let config = DataLoaderConfig::new(3);
        let mut loader = DataLoader::new(&ds, config);

        assert_eq!(loader.num_batches(), 0);
        assert!(loader.next_batch(&mut session).unwrap().is_none());
    }

    #[test]
    fn dataloader_single_sample() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(1, 3);
        let config = DataLoaderConfig::new(4);
        let mut loader = DataLoader::new(&ds, config);

        let batch = loader.next_batch(&mut session).unwrap().unwrap();
        let shape = session.tensor_shape(batch.input().unwrap()).unwrap();
        assert_eq!(shape, vec![1, 3]);

        assert!(loader.next_batch(&mut session).unwrap().is_none());
    }

    #[test]
    fn dataloader_batch_size_one() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(3, 2);
        let config = DataLoaderConfig::new(1);
        let mut loader = DataLoader::new(&ds, config);

        let mut count = 0;
        while let Some(batch) = loader.next_batch(&mut session).unwrap() {
            let shape = session.tensor_shape(batch.input().unwrap()).unwrap();
            assert_eq!(shape, vec![1, 2]);
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn dataloader_batch_size_larger_than_dataset() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(3, 2);
        let config = DataLoaderConfig::new(10);
        let mut loader = DataLoader::new(&ds, config);

        // Should get 1 batch of size 3
        let batch = loader.next_batch(&mut session).unwrap().unwrap();
        let shape = session.tensor_shape(batch.input().unwrap()).unwrap();
        assert_eq!(shape, vec![3, 2]);

        assert!(loader.next_batch(&mut session).unwrap().is_none());
    }

    #[test]
    fn dataloader_batch_size_larger_than_dataset_drop_last() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(3, 2);
        let config = DataLoaderConfig::new(10).with_drop_last(true);
        let mut loader = DataLoader::new(&ds, config);

        // drop_last: batch_size=10 > 3 samples, so no batches
        assert!(loader.next_batch(&mut session).unwrap().is_none());
    }

    #[test]
    fn dataloader_reset_restarts() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(4, 1);
        let config = DataLoaderConfig::new(2);
        let mut loader = DataLoader::new(&ds, config);

        // Consume all
        loader.next_batch(&mut session).unwrap();
        loader.next_batch(&mut session).unwrap();
        assert!(loader.next_batch(&mut session).unwrap().is_none());

        // Reset and iterate again
        loader.reset();
        let mut count = 0;
        while loader.next_batch(&mut session).unwrap().is_some() {
            count += 1;
        }
        assert_eq!(count, 2);
    }

    #[test]
    fn batch_accessor_methods() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(2, 3);
        let config = DataLoaderConfig::new(2);
        let mut loader = DataLoader::new(&ds, config);

        let batch = loader.next_batch(&mut session).unwrap().unwrap();
        assert!(batch.input().is_some());
        assert!(batch.target().is_some());
        assert!(batch.get("nonexistent").is_none());
    }

    #[test]
    fn data_item_single() {
        let item = DataItem::single("features", vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(item.tensors.len(), 1);
        assert_eq!(item.tensors[0].0, "features");
    }

    // ── Sampler Tests ────────────────────────────────────────────────

    #[test]
    fn sequential_sampler_indices() {
        let s = SequentialSampler::new(5);
        assert_eq!(s.len(), 5);
        assert_eq!(s.indices(), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn sequential_sampler_empty() {
        let s = SequentialSampler::new(0);
        assert!(s.is_empty());
        assert!(s.indices().is_empty());
    }

    #[test]
    fn random_sampler_is_permutation() {
        let s = RandomSampler::new(10).with_seed(42);
        let indices = s.indices();
        assert_eq!(indices.len(), 10);
        // All values 0..10 present (permutation)
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
        // Should not be sequential (with overwhelming probability)
        assert_ne!(indices, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn random_sampler_with_replacement() {
        let s = RandomSampler::new(5)
            .with_replacement(true)
            .with_num_samples(20)
            .with_seed(123);
        let indices = s.indices();
        assert_eq!(indices.len(), 20);
        // All indices should be in [0, 5)
        assert!(indices.iter().all(|&i| i < 5));
        // With 20 samples from 5 elements, we expect repeats
        let unique: std::collections::HashSet<usize> = indices.iter().copied().collect();
        assert!(unique.len() < 20);
    }

    #[test]
    fn random_sampler_repeated_passes_match_refill_reference_order() {
        fn refill_reference(size: usize, num_samples: usize, seed: u64) -> Vec<usize> {
            let mut rng = SimpleRng::new(seed);
            let mut result = Vec::with_capacity(num_samples);
            let mut idx: Vec<usize> = (0..size).collect();
            let full_passes = num_samples / size;
            for pass in 0..full_passes {
                if pass > 0 {
                    refill_sequential_indices(&mut idx);
                }
                rng.shuffle(&mut idx);
                result.extend_from_slice(&idx);
            }
            let remainder = num_samples % size;
            if remainder > 0 {
                if full_passes > 0 {
                    refill_sequential_indices(&mut idx);
                }
                rng.shuffle(&mut idx);
                result.extend_from_slice(&idx[..remainder]);
            }
            result
        }

        let size = 37;
        let num_samples = size * 9 + 13;
        let seed = 0xA5A5_5A5A_0123_4567;
        let sampler = RandomSampler::new(size)
            .with_num_samples(num_samples)
            .with_seed(seed);
        assert_eq!(
            sampler.indices(),
            refill_reference(size, num_samples, seed),
            "optimized repeated-pass sampler must preserve old refill order"
        );
    }

    #[test]
    fn subset_random_sampler_basic() {
        let subset = vec![2, 5, 8, 11];
        let s = SubsetRandomSampler::new(subset.clone()).with_seed(77);
        assert_eq!(s.len(), 4);
        let indices = s.indices();
        assert_eq!(indices.len(), 4);
        // All output indices come from the subset
        let mut sorted = indices.clone();
        sorted.sort();
        let mut expected = subset.clone();
        expected.sort();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn subset_random_sampler_empty() {
        let s = SubsetRandomSampler::new(Vec::new());
        assert!(s.is_empty());
        assert!(s.indices().is_empty());
    }

    #[test]
    fn weighted_random_sampler_favors_high_weight() {
        // Weight index 0 heavily, others very low
        let weights = vec![100.0, 0.1, 0.1, 0.1, 0.1];
        let s = WeightedRandomSampler::new(weights, 100).with_seed(42);
        let indices = s.indices().expect("weighted samples");
        assert_eq!(indices.len(), 100);
        // Most samples should be index 0
        let count_zero = indices.iter().filter(|&&i| i == 0).count();
        assert!(
            count_zero > 80,
            "expected most samples to be index 0, got {count_zero}/100"
        );
    }

    #[test]
    fn weighted_random_sampler_all_zero_weights() {
        let weights = vec![0.0, 0.0, 0.0];
        let s = WeightedRandomSampler::new(weights, 30).with_seed(99);
        assert!(s.indices().is_err());
    }

    #[test]
    fn weighted_random_sampler_rejects_negative_weights() {
        let weights = vec![-10.0, 5.0, 0.0];
        let s = WeightedRandomSampler::new(weights, 50).with_seed(7);
        assert!(s.indices().is_err());
    }

    #[test]
    fn weighted_random_sampler_rejects_nan_weights() {
        let weights = vec![f64::NAN, 5.0, 0.0];
        let s = WeightedRandomSampler::new(weights, 50).with_seed(7);
        assert!(s.indices().is_err());
    }

    #[test]
    fn batch_sampler_sequential() {
        let seq = SequentialSampler::new(10);
        let bs = BatchSampler::from_sequential(&seq, 3, false);
        let batches = bs.batches();
        assert_eq!(batches.len(), 4); // 3+3+3+1
        assert_eq!(batches[0], vec![0, 1, 2]);
        assert_eq!(batches[1], vec![3, 4, 5]);
        assert_eq!(batches[2], vec![6, 7, 8]);
        assert_eq!(batches[3], vec![9]);
    }

    #[test]
    fn batch_sampler_drop_last() {
        let seq = SequentialSampler::new(10);
        let bs = BatchSampler::from_sequential(&seq, 3, true);
        let batches = bs.batches();
        assert_eq!(batches.len(), 3); // drop the [9]
        assert_eq!(bs.len(), 3);
    }

    #[test]
    fn batch_sampler_from_random() {
        let rand = RandomSampler::new(8).with_seed(42);
        let bs = BatchSampler::from_random(&rand, 3, false);
        let batches = bs.batches();
        assert_eq!(batches.len(), 3); // 3+3+2
        // Total indices should be a permutation of 0..8
        let all: Vec<usize> = batches.into_iter().flatten().collect();
        assert_eq!(all.len(), 8);
        let mut sorted = all.clone();
        sorted.sort();
        assert_eq!(sorted, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn batch_sampler_batch_size_larger() {
        let seq = SequentialSampler::new(3);
        let bs = BatchSampler::from_sequential(&seq, 10, false);
        let batches = bs.batches();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0], vec![0, 1, 2]);
    }

    #[test]
    fn batch_sampler_batch_size_larger_drop_last() {
        let seq = SequentialSampler::new(3);
        let bs = BatchSampler::from_sequential(&seq, 10, true);
        assert!(bs.is_empty());
        assert!(bs.batches().is_empty());
    }

    #[test]
    fn dataloader_with_random_sampler_indices() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(6, 2);
        let sampler = RandomSampler::new(6).with_seed(42);
        let config = DataLoaderConfig::new(3);
        let mut loader = DataLoader::with_indices(&ds, sampler.indices(), config);

        let mut all_samples = Vec::new();
        while let Some(batch) = loader.next_batch(&mut session).unwrap() {
            let target = batch.target().unwrap();
            let vals = session.tensor_values(target).unwrap();
            all_samples.extend(vals);
        }
        // Should have all 6 targets (0..6) but in shuffled order
        assert_eq!(all_samples.len(), 6);
        let mut sorted: Vec<i64> = all_samples.iter().map(|&v| v as i64).collect();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn dataloader_rejects_out_of_range_sampler_index() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(2, 1);
        let config = DataLoaderConfig::new(2);
        let mut loader = DataLoader::with_indices(&ds, vec![0, 2], config);

        let message = loader
            .next_batch(&mut session)
            .err()
            .map(|err| err.to_string())
            .unwrap_or_default();
        assert!(
            message.contains("sampler index out of range"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn dataloader_with_subset_sampler_train_val_split() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(10, 1);

        // Train on first 7, val on last 3
        let train_sampler = SubsetRandomSampler::new((0..7).collect()).with_seed(11);
        let val_sampler = SubsetRandomSampler::new((7..10).collect()).with_seed(22);

        let train_config = DataLoaderConfig::new(7);
        let val_config = DataLoaderConfig::new(3);

        let mut train_loader = DataLoader::with_indices(&ds, train_sampler.indices(), train_config);
        let mut val_loader = DataLoader::with_indices(&ds, val_sampler.indices(), val_config);

        let train_batch = train_loader.next_batch(&mut session).unwrap().unwrap();
        let train_targets = session
            .tensor_values(train_batch.target().unwrap())
            .unwrap();
        // All train targets should be in 0..7
        assert!(train_targets.iter().all(|&v| (v as usize) < 7));

        let val_batch = val_loader.next_batch(&mut session).unwrap().unwrap();
        let val_targets = session.tensor_values(val_batch.target().unwrap()).unwrap();
        // All val targets should be in 7..10
        assert!(
            val_targets
                .iter()
                .all(|&v| (v as usize) >= 7 && (v as usize) < 10)
        );
    }

    // ── Transform Tests ─────────────────────────────────────────────────

    #[test]
    fn normalize_transform() {
        let item = DataItem::single("input", vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]);
        let t = NormalizeTransform::new("input", vec![10.0, 30.0], vec![10.0, 10.0])
            .expect("transform");
        let result = t.apply(item);
        let vals = &result.tensors[0].1;
        // Channel 0: (10-10)/10=0, (20-10)/10=1
        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[1] - 1.0).abs() < 1e-10);
        // Channel 1: (30-30)/10=0, (40-30)/10=1
        assert!((vals[2] - 0.0).abs() < 1e-10);
        assert!((vals[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn normalize_transform_respects_declared_channel_dimension() {
        let item = DataItem::single("input", vec![10.0, 20.0, 30.0, 40.0], vec![1, 2, 2]);
        let t = NormalizeTransform::new("input", vec![10.0, 30.0], vec![10.0, 10.0])
            .expect("transform");

        let result = t.apply(item);

        assert_eq!(
            result.tensors[0].1,
            vec![10.0, 20.0, 30.0, 40.0],
            "stats for two channels must not be applied to a one-channel tensor"
        );
    }

    #[test]
    fn normalize_transform_rejects_mismatched_stats() {
        let err =
            NormalizeTransform::new("input", vec![0.0, 1.0], vec![1.0]).expect_err("mismatch");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet { .. }
            ))
        ));
        if let AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
            ft_dispatch::DispatchKeyError::IncompatibleSet { reason },
        )) = err
        {
            assert_eq!(
                reason,
                "NormalizeTransform requires mean/std lengths to match"
            );
        }
    }

    #[test]
    fn normalize_transform_rejects_non_positive_std() {
        let err = NormalizeTransform::new("input", vec![0.0], vec![0.0]).expect_err("zero std");
        assert!(matches!(
            err,
            AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                ft_dispatch::DispatchKeyError::IncompatibleSet { .. }
            ))
        ));
        if let AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
            ft_dispatch::DispatchKeyError::IncompatibleSet { reason },
        )) = err
        {
            assert_eq!(reason, "NormalizeTransform requires finite std values > 0");
        }
    }

    #[test]
    fn scale_transform() {
        let item = DataItem::single("input", vec![1.0, 2.0, 3.0], vec![3]);
        let t = ScaleTransform::new("input", 2.0);
        let result = t.apply(item);
        assert_eq!(result.tensors[0].1, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn lambda_transform() {
        let item = DataItem::single("input", vec![1.0, 2.0], vec![2]);
        let t = LambdaTransform::new(|mut item| {
            for (_, vals, _) in &mut item.tensors {
                for v in vals.iter_mut() {
                    *v += 100.0;
                }
            }
            item
        });
        let result = t.apply(item);
        assert_eq!(result.tensors[0].1, vec![101.0, 102.0]);
    }

    #[test]
    fn compose_transforms() {
        let item = DataItem::single("input", vec![1.0, 2.0], vec![2]);
        let t = Compose::new(vec![
            Box::new(ScaleTransform::new("input", 10.0)),
            Box::new(ScaleTransform::new("input", 0.5)),
        ]);
        let result = t.apply(item);
        // 1*10*0.5=5, 2*10*0.5=10
        assert_eq!(result.tensors[0].1, vec![5.0, 10.0]);
    }

    #[test]
    fn transform_dataset_wraps() {
        let ds = TensorDataset::new(vec![
            DataItem::single("input", vec![1.0, 2.0], vec![2]),
            DataItem::single("input", vec![3.0, 4.0], vec![2]),
        ]);
        let tds = TransformDataset::new(ds, Box::new(ScaleTransform::new("input", 3.0)));
        assert_eq!(tds.len(), 2);
        let item0 = tds.get(0);
        assert_eq!(item0.tensors[0].1, vec![3.0, 6.0]);
        let item1 = tds.get(1);
        assert_eq!(item1.tensors[0].1, vec![9.0, 12.0]);
    }

    #[test]
    fn normalize_wrong_name_passthrough() {
        let item = DataItem::single("input", vec![5.0], vec![1]);
        let t = NormalizeTransform::new("other", vec![0.0], vec![1.0]).expect("transform");
        let result = t.apply(item);
        // "input" not affected because name doesn't match
        assert_eq!(result.tensors[0].1, vec![5.0]);
    }

    #[test]
    fn dataloader_rejects_mismatched_tensor_names() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = TensorDataset::new(vec![
            DataItem::single("input", vec![1.0, 2.0], vec![2]),
            DataItem::single("features", vec![3.0, 4.0], vec![2]),
        ]);
        let config = DataLoaderConfig::new(2);
        let mut loader = DataLoader::new(&ds, config);

        let message = loader
            .next_batch(&mut session)
            .err()
            .map(|err| err.to_string())
            .unwrap_or_default();
        assert!(
            message.contains("tensor names differ"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn dataloader_rejects_shape_value_length_mismatch() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = TensorDataset::new(vec![
            DataItem::single("input", vec![1.0], vec![2]),
            DataItem::single("input", vec![3.0, 4.0], vec![2]),
        ]);
        let config = DataLoaderConfig::new(2);
        let mut loader = DataLoader::new(&ds, config);

        let message = loader
            .next_batch(&mut session)
            .err()
            .map(|err| err.to_string())
            .unwrap_or_default();
        assert!(
            message.contains("values length does not match declared tensor shape"),
            "unexpected error: {message}"
        );
    }

    // ── Subset and random_split tests ──────────────────────────────────

    #[test]
    fn subset_basic() {
        let ds = std::sync::Arc::new(make_dataset(10, 2));
        let subset = Subset::new(ds, vec![0, 3, 7]);
        assert_eq!(subset.len(), 3);
        assert_eq!(subset.indices(), &[0, 3, 7]);
        let item = subset.get(1); // index 1 in subset = index 3 in original
        assert!(!item.tensors.is_empty());
    }

    #[test]
    fn random_split_lengths_sum() {
        let ds = make_dataset(100, 2);
        let splits = random_split(ds, &[70, 20, 10], 42);
        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].len(), 70);
        assert_eq!(splits[1].len(), 20);
        assert_eq!(splits[2].len(), 10);
    }

    #[test]
    fn random_split_no_overlap() {
        let ds = make_dataset(20, 1);
        let splits = random_split(ds, &[10, 10], 123);
        let mut all_indices: Vec<usize> = Vec::new();
        all_indices.extend_from_slice(splits[0].indices());
        all_indices.extend_from_slice(splits[1].indices());
        all_indices.sort_unstable();
        all_indices.dedup();
        assert_eq!(
            all_indices.len(),
            20,
            "splits should have no overlapping indices"
        );
    }

    #[test]
    fn random_split_deterministic() {
        let ds1 = make_dataset(50, 1);
        let ds2 = make_dataset(50, 1);
        let splits1 = random_split(ds1, &[30, 20], 42);
        let splits2 = random_split(ds2, &[30, 20], 42);
        assert_eq!(splits1[0].indices(), splits2[0].indices());
        assert_eq!(splits1[1].indices(), splits2[1].indices());
    }

    #[test]
    #[should_panic(expected = "sum of lengths")]
    fn random_split_rejects_too_large() {
        let ds = make_dataset(5, 1);
        random_split(ds, &[3, 4], 0);
    }

    #[test]
    #[should_panic(expected = "sum of lengths")]
    fn random_split_rejects_too_small() {
        let ds = make_dataset(5, 1);
        random_split(ds, &[2, 2], 0);
    }

    // ── frankentorch-4i2: Sampler builder and edge case tests ──────────

    #[test]
    fn random_sampler_deterministic_with_same_seed() {
        let a = RandomSampler::new(20).with_seed(999).indices();
        let b = RandomSampler::new(20).with_seed(999).indices();
        assert_eq!(a, b, "same seed must produce identical indices");
    }

    #[test]
    fn random_sampler_different_seeds_differ() {
        let a = RandomSampler::new(20).with_seed(1).indices();
        let b = RandomSampler::new(20).with_seed(2).indices();
        assert_ne!(
            a, b,
            "different seeds should produce different permutations"
        );
    }

    #[test]
    fn random_sampler_truncated_without_replacement() {
        let s = RandomSampler::new(10).with_num_samples(5).with_seed(42);
        let indices = s.indices();
        assert_eq!(indices.len(), 5);
        assert_eq!(s.len(), 5);
        // All indices in range and unique (no replacement)
        let unique: std::collections::HashSet<usize> = indices.iter().copied().collect();
        assert_eq!(unique.len(), 5);
        assert!(indices.iter().all(|&i| i < 10));
    }

    #[test]
    fn random_sampler_empty() {
        let s = RandomSampler::new(0);
        assert!(s.is_empty());
        assert!(s.indices().is_empty());
    }

    #[test]
    fn random_sampler_empty_with_replacement_returns_empty() {
        let s = RandomSampler::new(0)
            .with_replacement(true)
            .with_num_samples(5)
            .with_seed(7);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert!(s.indices().is_empty());
    }

    #[test]
    fn random_sampler_num_samples_repeats_shuffled_passes_without_replacement() {
        let s = RandomSampler::new(3).with_num_samples(5).with_seed(123);
        let indices = s.indices();
        assert_eq!(indices.len(), 5);
        assert_eq!(s.len(), 5);
        assert_eq!(indices, vec![2, 0, 1, 1, 2]);
        assert!(indices.iter().all(|&i| i < 3));

        let mut first_pass = indices[..3].to_vec();
        first_pass.sort();
        assert_eq!(first_pass, vec![0, 1, 2]);

        let remainder: std::collections::HashSet<usize> = indices[3..].iter().copied().collect();
        assert_eq!(remainder.len(), 2);

        let unique: std::collections::HashSet<usize> = indices.iter().copied().collect();
        assert_eq!(unique.len(), 3);
    }

    #[test]
    fn random_sampler_repeated_passes_preserve_exact_order() {
        let s = RandomSampler::new(7).with_num_samples(19).with_seed(123);
        assert_eq!(
            s.indices(),
            vec![6, 3, 0, 4, 5, 2, 1, 2, 4, 5, 3, 6, 0, 1, 2, 4, 6, 5, 3]
        );
    }

    #[test]
    fn weighted_random_sampler_deterministic() {
        let w = vec![1.0, 2.0, 3.0];
        let a = WeightedRandomSampler::new(w.clone(), 50)
            .with_seed(42)
            .indices()
            .expect("weighted samples");
        let b = WeightedRandomSampler::new(w, 50)
            .with_seed(42)
            .indices()
            .expect("weighted samples");
        assert_eq!(a, b, "same seed must produce identical weighted samples");
    }

    #[test]
    fn weighted_random_sampler_two_weight_fast_path_preserves_order() {
        let indices = WeightedRandomSampler::new(vec![1.0, 3.0], 32)
            .with_seed(0x5151_0002)
            .indices()
            .expect("weighted samples");
        assert_eq!(
            indices,
            vec![
                1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0,
                0, 1, 1, 0,
            ]
        );
    }

    #[test]
    fn weighted_random_sampler_three_weight_fast_path_preserves_order() {
        let indices = WeightedRandomSampler::new(vec![1.0, 3.0, 6.0], 32)
            .with_seed(0x5151_0003)
            .indices()
            .expect("weighted samples");
        assert_eq!(
            indices,
            vec![
                2, 1, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 2, 2,
                0, 2, 1, 2,
            ]
        );
    }

    #[test]
    fn weighted_random_sampler_four_weight_fast_path_preserves_order() {
        let indices = WeightedRandomSampler::new(vec![1.0, 2.0, 3.0, 4.0], 32)
            .with_seed(0x5151_0004)
            .indices()
            .expect("weighted samples");
        assert_eq!(
            indices,
            vec![
                3, 2, 3, 2, 3, 2, 3, 3, 3, 3, 2, 2, 2, 3, 2, 2, 3, 0, 1, 2, 1, 3, 3, 2, 3, 2, 3, 2,
                3, 3, 1, 2,
            ]
        );
    }

    fn weighted_sampler_branchless_golden_summary() -> String {
        let two = WeightedRandomSampler::new(vec![1.0, 3.0], 32)
            .with_seed(0x5151_0002)
            .indices()
            .expect("two-weight samples");
        let three = WeightedRandomSampler::new(vec![1.0, 3.0, 6.0], 32)
            .with_seed(0x5151_0003)
            .indices()
            .expect("three-weight samples");
        let four = WeightedRandomSampler::new(vec![1.0, 2.0, 3.0, 4.0], 32)
            .with_seed(0x5151_0004)
            .indices()
            .expect("four-weight samples");
        let edge_inputs = [0.0, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875];
        let edges: Vec<usize> = edge_inputs
            .into_iter()
            .map(|u| threshold_index_4(u, 0.25, 0.5, 0.75))
            .collect();

        format!(
            "weighted_sampler_branchless_frankentorch-sx6l\n\
             two_seed=0x51510002 weights=[1.0, 3.0] samples={two:?}\n\
             three_seed=0x51510003 weights=[1.0, 3.0, 6.0] samples={three:?}\n\
             four_seed=0x51510004 weights=[1.0, 2.0, 3.0, 4.0] samples={four:?}\n\
             threshold4_edges inputs=[0.0, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875] outputs={edges:?}\n"
        )
    }

    #[test]
    fn weighted_sampler_branchless_golden_output_matches_fixture() {
        assert_eq!(
            weighted_sampler_branchless_golden_summary(),
            include_str!(
                "../../../artifacts/optimization/golden_outputs/ft_data_weighted_sampler_branchless_frankentorch-sx6l.txt"
            )
        );
    }

    fn weighted_sampler_large_cardinality_golden_summary() -> String {
        let weights: Vec<f64> = (1..=4096).map(|i| f64::from((i % 17) + 1)).collect();
        let samples = WeightedRandomSampler::new(weights, 32)
            .with_seed(0x5151_4096)
            .indices()
            .expect("large-cardinality weighted samples");

        format!(
            "weighted_sampler_large_cardinality_frankentorch-j54u\n\
             seed=0x51514096 weights=(1..=4096).map((i % 17) + 1) samples={samples:?}\n"
        )
    }

    #[test]
    fn weighted_sampler_large_cardinality_golden_output_matches_fixture() {
        assert_eq!(
            weighted_sampler_large_cardinality_golden_summary(),
            include_str!(
                "../../../artifacts/optimization/golden_outputs/ft_data_weighted_large_cardinality_frankentorch-j54u.txt"
            )
        );
    }

    #[test]
    fn weighted_sampler_bucket_index_matches_binary_reference() {
        let weights: Vec<f64> = (1..=4096).map(|i| f64::from((i % 17) + 1)).collect();
        let total: f64 = weights.iter().sum();
        let mut cumulative = Vec::with_capacity(weights.len());
        let mut running = 0.0;
        for weight in weights {
            running += weight;
            cumulative.push(running / total);
        }

        let bucket_index =
            ThresholdBucketIndex::new(&cumulative).expect("positive weights should be indexed");
        let mut rng = SimpleRng::new(0x5151_4096);
        for _ in 0..10_000 {
            let sample = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            assert_eq!(
                bucket_index.find(&cumulative, sample),
                threshold_binary_index(&cumulative, sample),
                "sample={sample}"
            );
        }

        for &sample in &[
            0.0,
            cumulative[0],
            cumulative[17],
            cumulative[2048],
            cumulative[4094],
            0.999_999,
        ] {
            assert_eq!(
                bucket_index.find(&cumulative, sample),
                threshold_binary_index(&cumulative, sample),
                "edge sample={sample}"
            );
        }
    }

    #[test]
    fn weighted_sampler_four_threshold_helper_matches_branch_chain_edges() {
        let first = 0.25;
        let second = 0.5;
        let third = 0.75;
        let samples = [0.0, first, 0.375, second, 0.625, third, 0.875];

        for u in samples {
            let expected = if u <= first {
                0
            } else if u <= second {
                1
            } else if u <= third {
                2
            } else {
                3
            };
            assert_eq!(threshold_index_4(u, first, second, third), expected);
        }
    }

    #[test]
    fn weighted_sampler_finite_threshold_cmp_matches_total_cmp_edges() {
        let thresholds: [f64; 6] = [0.0, 0.25, 0.25, 0.5, 0.875, 1.0];
        let samples: [f64; 7] = [0.0, 0.25, 0.375, 0.5, 0.625, 0.875, 0.999_999];

        for sample in samples {
            let expected = thresholds.binary_search_by(|threshold| threshold.total_cmp(&sample));
            let actual =
                thresholds.binary_search_by(|threshold| finite_threshold_cmp(*threshold, sample));
            assert_eq!(actual, expected, "sample={sample}");
        }
    }

    #[test]
    fn weighted_random_sampler_empty_weights() {
        let s = WeightedRandomSampler::new(Vec::new(), 10).with_seed(42);
        assert_eq!(s.len(), 10);
        assert!(s.indices().is_err());
    }

    #[test]
    fn weighted_random_sampler_single_weight() {
        let s = WeightedRandomSampler::new(vec![5.0], 20).with_seed(42);
        let indices = s.indices().expect("weighted samples");
        assert_eq!(indices.len(), 20);
        assert!(
            indices.iter().all(|&i| matches!(i, 0)),
            "single weight = always index 0"
        );
    }

    #[test]
    fn weighted_random_sampler_single_weight_preserves_validation_order() {
        let zero_samples = WeightedRandomSampler::new(vec![5.0], 0).with_seed(42);
        let zero_samples_message = zero_samples
            .indices()
            .expect_err("zero samples should fail before single-weight fast path")
            .to_string();
        assert!(
            zero_samples_message.contains("num_samples must be > 0"),
            "unexpected error: {zero_samples_message}"
        );

        let infinity = WeightedRandomSampler::new(vec![f64::INFINITY], 8).with_seed(42);
        let infinity_message = infinity
            .indices()
            .expect_err("infinite single weight must fail before fast path")
            .to_string();
        assert!(
            infinity_message.contains("weights must be finite"),
            "unexpected error: {infinity_message}"
        );

        let zero_weight = WeightedRandomSampler::new(vec![0.0], 8).with_seed(42);
        let zero_weight_message = zero_weight
            .indices()
            .expect_err("zero single weight must fail before fast path")
            .to_string();
        assert!(
            zero_weight_message.contains("sum of weights must be > 0"),
            "unexpected error: {zero_weight_message}"
        );
    }

    #[test]
    fn weighted_random_sampler_rejects_infinity_weight() {
        let s = WeightedRandomSampler::new(vec![f64::INFINITY, 1.0], 50).with_seed(42);
        assert!(s.indices().is_err());
    }

    #[test]
    fn weighted_random_sampler_rejects_overflowed_weight_sum() {
        let s = WeightedRandomSampler::new(vec![f64::MAX, f64::MAX], 16).with_seed(42);
        let message = s
            .indices()
            .expect_err("overflowed total weight must fail closed")
            .to_string();
        assert!(
            message.contains("sum of weights must be finite"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn weighted_random_sampler_rejects_mixed_invalid_weight_sequence() {
        let s = WeightedRandomSampler::new(vec![1.0, f64::NAN, -5.0, 3.0, f64::INFINITY, 2.0], 16)
            .with_seed(123);

        assert!(s.indices().is_err());
    }

    #[test]
    fn weighted_random_sampler_zero_num_samples() {
        let s = WeightedRandomSampler::new(vec![1.0, 2.0], 0).with_seed(42);
        assert!(s.is_empty());
        assert!(s.indices().is_err());
    }

    #[test]
    fn subset_random_sampler_deterministic() {
        let indices = vec![10, 20, 30, 40, 50];
        let a = SubsetRandomSampler::new(indices.clone())
            .with_seed(77)
            .indices();
        let b = SubsetRandomSampler::new(indices).with_seed(77).indices();
        assert_eq!(a, b, "same seed must produce identical shuffle");
    }

    #[test]
    fn batch_sampler_from_subset() {
        let subset = SubsetRandomSampler::new(vec![0, 1, 2, 3, 4]).with_seed(42);
        let bs = BatchSampler::from_subset(&subset, 2, false);
        let batches = bs.batches();
        assert_eq!(batches.len(), 3); // ceil(5/2) = 3
        let all: Vec<usize> = batches.into_iter().flatten().collect();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn batch_sampler_empty_indices() {
        let bs = BatchSampler::new(Vec::new(), 3, false);
        assert!(bs.is_empty());
        assert_eq!(bs.len(), 0);
        assert!(bs.batches().is_empty());
    }

    #[test]
    fn batch_sampler_batch_size_zero() {
        let bs = BatchSampler::new(vec![0, 1, 2], 0, false);
        assert_eq!(bs.len(), 0);
        assert!(bs.batches().is_empty());
    }

    #[test]
    fn batch_sampler_exact_division() {
        let seq = SequentialSampler::new(9);
        let bs = BatchSampler::from_sequential(&seq, 3, false);
        assert_eq!(bs.len(), 3);
        let batches = bs.batches();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 3);
        assert_eq!(batches[2].len(), 3);
    }

    #[test]
    fn batch_sampler_exact_division_drop_last_same() {
        let seq = SequentialSampler::new(9);
        let bs_keep = BatchSampler::from_sequential(&seq, 3, false);
        let bs_drop = BatchSampler::from_sequential(&seq, 3, true);
        // When evenly divisible, drop_last makes no difference
        assert_eq!(bs_keep.len(), bs_drop.len());
        assert_eq!(bs_keep.batches(), bs_drop.batches());
    }

    #[test]
    fn dataloader_config_builder_chain() {
        let config = DataLoaderConfig::new(32)
            .with_shuffle(true)
            .with_drop_last(true);
        assert_eq!(config.batch_size, 32);
        assert!(config.shuffle);
        assert!(config.drop_last);
    }

    #[test]
    fn dataloader_with_weighted_sampler() {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let ds = make_dataset(5, 1);
        // Heavily weight index 0
        let sampler = WeightedRandomSampler::new(vec![100.0, 0.0, 0.0, 0.0, 0.0], 10).with_seed(42);
        let config = DataLoaderConfig::new(10);
        let mut loader =
            DataLoader::with_indices(&ds, sampler.indices().expect("weighted samples"), config);

        let batch = loader.next_batch(&mut session).unwrap().unwrap();
        let targets = session.tensor_values(batch.target().unwrap()).unwrap();
        // All targets should be 0.0 since only index 0 has weight
        assert!(
            targets.iter().all(|&v| (v - 0.0).abs() < 1e-10),
            "weighted sampler should only produce index 0, got targets: {targets:?}"
        );
    }

    #[test]
    #[should_panic(expected = "must equal dataset size")]
    fn random_split_rejects_partial_sum() {
        // PyTorch requires random_split lengths to exhaust the dataset.
        let ds = make_dataset(20, 1);
        random_split(ds, &[5, 5], 42);
    }
}
