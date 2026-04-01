#![forbid(unsafe_code)]

use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, TensorNodeId};

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
}

// ── TensorDataset ────────────────────────────────────────────────────────

/// A dataset backed by a collection of pre-loaded tensor data.
///
/// Each sample is a `DataItem` containing one or more named tensors.
/// This is equivalent to PyTorch's `TensorDataset`.
pub struct TensorDataset {
    items: Vec<DataItem>,
}

impl TensorDataset {
    /// Create a new `TensorDataset` from a vector of data items.
    pub fn new(items: Vec<DataItem>) -> Self {
        Self { items }
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
        let items = inputs
            .into_iter()
            .zip(targets)
            .map(|((iv, is), (tv, ts))| DataItem::input_target(iv, is, tv, ts))
            .collect();
        Self { items }
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, index: usize) -> DataItem {
        self.items[index].clone()
    }
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
        self.num_samples
    }

    pub fn is_empty(&self) -> bool {
        self.num_samples == 0
    }

    pub fn indices(&self) -> Vec<usize> {
        let mut rng = SimpleRng::new(self.seed);
        if self.replacement {
            (0..self.num_samples)
                .map(|_| rng.next_below(self.size))
                .collect()
        } else {
            let mut idx: Vec<usize> = (0..self.size).collect();
            rng.shuffle(&mut idx);
            idx.truncate(self.num_samples);
            idx
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

    pub fn indices(&self) -> Vec<usize> {
        if self.weights.is_empty() {
            return Vec::new();
        }

        // Build cumulative distribution
        let total: f64 = self.weights.iter().sum();
        if total <= 0.0 {
            // All-zero weights: uniform sampling
            let mut rng = SimpleRng::new(self.seed);
            return (0..self.num_samples)
                .map(|_| rng.next_below(self.weights.len()))
                .collect();
        }

        let mut cumulative = Vec::with_capacity(self.weights.len());
        let mut running = 0.0;
        for &w in &self.weights {
            running += w.max(0.0);
            cumulative.push(running / total);
        }

        let mut rng = SimpleRng::new(self.seed);
        let mut result = Vec::with_capacity(self.num_samples);
        for _ in 0..self.num_samples {
            // Generate uniform [0, 1) using the RNG
            let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            // Binary search for the index
            let idx = match cumulative.binary_search_by(|c| c.partial_cmp(&u).unwrap()) {
                Ok(i) => i,
                Err(i) => i.min(self.weights.len() - 1),
            };
            result.push(idx);
        }
        result
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
    /// Simple LCG state for shuffling (deterministic, no external deps).
    rng_state: u64,
}

impl<'a, D: Dataset> DataLoader<'a, D> {
    /// Create a new `DataLoader` for the given dataset and config.
    pub fn new(dataset: &'a D, config: DataLoaderConfig) -> Self {
        let n = dataset.len();
        let indices: Vec<usize> = (0..n).collect();
        Self {
            dataset,
            config,
            indices,
            position: 0,
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
            rng_state: 0xDEAD_BEEF_CAFE_1234,
        }
    }

    /// Set the random seed for shuffling.
    pub fn seed(mut self, seed: u64) -> Self {
        self.rng_state = seed;
        self
    }

    /// Reset the loader to the beginning of the dataset.
    /// If shuffle is enabled, re-shuffles the indices.
    pub fn reset(&mut self) {
        self.position = 0;
        if self.config.shuffle {
            self.shuffle_indices();
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

        // Collect samples for this batch
        let mut samples = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let idx = self.indices[self.position + i];
            samples.push(self.dataset.get(idx));
        }
        self.position += batch_size;

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
        let sample_numel: usize = first_shape.iter().product();
        for sample in samples.iter().skip(1) {
            if sample.tensors.len() != num_tensors {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "DataLoader: samples have different numbers of tensors",
                    },
                )));
            }
            let (_, _, ref s_shape) = sample.tensors[tensor_idx];
            if s_shape != first_shape {
                return Err(AutogradError::Dispatch(ft_dispatch::DispatchError::Key(
                    ft_dispatch::DispatchKeyError::IncompatibleSet {
                        reason: "DataLoader: tensor shapes differ across samples in batch",
                    },
                )));
            }
        }

        // Stack along batch dimension: new shape = [batch_size] ++ sample_shape
        let mut batched_values = Vec::with_capacity(batch_size * sample_numel);
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
pub struct NormalizeTransform {
    /// Name of the tensor to normalize (e.g., "input").
    tensor_name: String,
    /// Per-channel means.
    mean: Vec<f64>,
    /// Per-channel standard deviations.
    std: Vec<f64>,
}

impl NormalizeTransform {
    pub fn new(tensor_name: &str, mean: Vec<f64>, std: Vec<f64>) -> Self {
        Self {
            tensor_name: tensor_name.to_string(),
            mean,
            std,
        }
    }
}

impl Transform for NormalizeTransform {
    fn apply(&self, mut item: DataItem) -> DataItem {
        for (name, values, _shape) in &mut item.tensors {
            if name == &self.tensor_name && !self.mean.is_empty() {
                let channels = self.mean.len();
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
            transform: transform,
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
        let indices = s.indices();
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
        // All-zero weights: uniform sampling
        let weights = vec![0.0, 0.0, 0.0];
        let s = WeightedRandomSampler::new(weights, 30).with_seed(99);
        let indices = s.indices();
        assert_eq!(indices.len(), 30);
        assert!(indices.iter().all(|&i| i < 3));
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
        let t = NormalizeTransform::new("input", vec![10.0, 30.0], vec![10.0, 10.0]);
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
        let t = NormalizeTransform::new("other", vec![0.0], vec![1.0]);
        let result = t.apply(item);
        // "input" not affected because name doesn't match
        assert_eq!(result.tensors[0].1, vec![5.0]);
    }
}
