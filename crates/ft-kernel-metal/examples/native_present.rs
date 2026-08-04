use ft_kernel_metal::{
    Error,
    compute::Gateway,
    presentation::{
        NativePresenter, PresentOutcome, PresentationConfig, PresentationError, PresentationState,
    },
};
use std::{
    error::Error as StdError,
    io, thread,
    time::{Duration, Instant},
};

fn main() -> Result<(), Box<dyn StdError>> {
    let gateway = match Gateway::open() {
        Ok(gateway) => gateway,
        Err(Error::Unavailable) => {
            eprintln!("native Metal presentation is unavailable on this target");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };

    let (width, height) = (960_u32, 540_u32);
    let pixels = gradient(width, height);
    let surface = gateway.buffer_u32(&pixels)?;
    let mut presenter = NativePresenter::open(
        &gateway,
        PresentationConfig::new(width, height, "FrankenTorch native Metal presentation"),
    )?;

    let deadline = Instant::now() + Duration::from_secs(2);
    let mut presented = 0_u32;
    while Instant::now() < deadline {
        match presenter.present_rgba8(&surface, width, height, width as usize * 4) {
            Ok(PresentOutcome::Presented) => presented += 1,
            Ok(PresentOutcome::Occluded) => {}
            Err(PresentationError::Closed) => break,
            Err(error) => return Err(error.into()),
        }
        if presenter.poll_events()? == PresentationState::Closed {
            break;
        }
        thread::sleep(Duration::from_millis(16));
    }
    presenter.close()?;

    if presented == 0 {
        return Err(io::Error::other(
            "native preview acquired no drawable during the smoke window",
        )
        .into());
    }
    println!(
        "presented {presented} RGBA8 frames on {} without host pixel readback",
        gateway.device_name()
    );
    Ok(())
}

fn gradient(width: u32, height: u32) -> Vec<u32> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            let red = (x * 255 / width.saturating_sub(1)) as u8;
            let green = (y * 255 / height.saturating_sub(1)) as u8;
            let blue = 255_u8.saturating_sub(red / 2).saturating_sub(green / 2);
            pixels.push(u32::from_le_bytes([red, green, blue, 255]));
        }
    }
    pixels
}
