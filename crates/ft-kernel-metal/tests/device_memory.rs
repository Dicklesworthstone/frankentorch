//! The device's memory figures: `Gateway::recommended_working_set_bytes`
//! and `Gateway::allocated_bytes`. One test in a process of its own, so
//! nothing else allocates on the device while it measures.

use ft_kernel_metal::Error;
use ft_kernel_metal::compute::Gateway;

/// More than any driver's rounding of one buffer: Apple silicon's Metal
/// rounds a buffer of 16 KiB or more up to its 16 KiB pages, and takes a
/// smaller one from a pool it already holds.
const ROUNDING_BYTES: u64 = 64 << 10;

#[test]
fn the_device_counts_a_buffer_while_it_lives() -> Result<(), Error> {
    let gateway = match Gateway::open() {
        Ok(gateway) => gateway,
        Err(Error::Unavailable) => {
            println!("no Metal device: skipped");
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    let recommended = gateway.recommended_working_set_bytes();
    println!("recommended working set {recommended} bytes");
    assert!(recommended > 0, "a device recommends a working set");
    for bytes in [1_usize, 4_097, 16_384, 16_385, 1 << 20, 64 << 20] {
        let wide = u64::try_from(bytes).unwrap_or(u64::MAX);
        let before = gateway.allocated_bytes();
        let during = {
            let _buffer = gateway.buffer_zeroed(bytes)?;
            let during = gateway.allocated_bytes();
            // Every gateway names the one device.
            assert_eq!(Gateway::open()?.allocated_bytes(), during);
            during
        };
        let after = gateway.allocated_bytes();
        let grew = during.saturating_sub(before);
        println!("{bytes} bytes: the device held {grew} more while it lived");
        if wide >= ROUNDING_BYTES {
            assert!(
                grew >= wide && grew < wide + ROUNDING_BYTES,
                "a buffer of {bytes} bytes grew the device by {grew}"
            );
        } else {
            assert!(
                grew < 2 * ROUNDING_BYTES,
                "a buffer of {bytes} bytes grew the device by {grew}"
            );
        }
        assert_eq!(after, before, "releasing {bytes} bytes frees them");
    }
    Ok(())
}
