//! Native input through a real AppKit window.
//!
//! Each case posts real `NSEvent`s (and one Core Graphics scroll event) for
//! the presenter's own window into the application queue. They reach the
//! presenter through the same `nextEventMatchingMask` pump, `sendEvent:`
//! dispatch and translation as user input, so this qualifies the typed event
//! path end to end. Focus changes activate and deactivate the application.
//!
//! The window must be opened on the process main thread, so this target runs
//! without the libtest harness:
//!
//! ```text
//! cargo test -p ft-kernel-metal --features synthetic-input --test native_input
//! ```
//!
//! A real pointer moving during the run can add its own motion events; the
//! assertions read the events next to each case's own markers, so stray
//! motion cannot satisfy or fail them.

#[cfg(target_os = "macos")]
fn main() {
    macos::main();
}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("native_input: skipped (native presentation is macOS-only)");
}

#[cfg(target_os = "macos")]
mod macos {
    use ft_kernel_metal::compute::{Gateway, SharedBuffer};
    use ft_kernel_metal::presentation::{
        ButtonSet, InputEvent, LogicalPoint, Modifiers, NativePresenter, PixelDensity,
        PointerButton, PresentationConfig, PresentationError, PresentationState, ScrollUnit,
        SurfaceMetrics, SyntheticInput,
    };
    use std::time::{Duration, Instant};

    type Check = Result<(), String>;
    type Case = (&'static str, fn(&mut NativePresenter) -> Check);

    const WIDTH: u32 = 320;
    const HEIGHT: u32 = 200;
    /// kVK_ANSI_A, kVK_Space and kVK_Escape.
    const KEY_A: u16 = 0;
    const KEY_SPACE: u16 = 49;
    const KEY_ESCAPE: u16 = 53;

    pub(super) fn main() {
        let gateway = match Gateway::open() {
            Ok(gateway) => gateway,
            Err(error) => {
                println!("native_input: skipped (no Metal gateway: {error})");
                return;
            }
        };
        let config = PresentationConfig::new(WIDTH, HEIGHT, "ft-kernel-metal native input");
        let mut presenter = match NativePresenter::open(&gateway, config) {
            Ok(presenter) => presenter,
            Err(PresentationError::Unavailable) => {
                println!("native_input: skipped (native presentation unavailable)");
                return;
            }
            Err(error) => {
                println!("native_input: FAILED to open the presenter: {error}");
                std::process::exit(1);
            }
        };
        let cases: [Case; 11] = [
            ("initial_focus_and_geometry", initial_focus_and_geometry),
            (
                "pointer_motion_is_logical_top_left",
                pointer_motion_is_logical_top_left,
            ),
            ("primary_press_drag_release", primary_press_drag_release),
            (
                "secondary_click_carries_modifiers",
                secondary_click_carries_modifiers,
            ),
            (
                "keys_carry_code_text_repeat_and_modifiers",
                keys_carry_code_text_repeat_and_modifiers,
            ),
            ("modifier_changes_are_typed", modifier_changes_are_typed),
            (
                "pixel_scroll_reaches_the_window",
                pixel_scroll_reaches_the_window,
            ),
            (
                "poll_events_keeps_input_queued",
                poll_events_keeps_input_queued,
            ),
            ("resize_reports_new_geometry", resize_reports_new_geometry),
            (
                "a_user_resize_leaves_the_drawable",
                a_user_resize_leaves_the_drawable,
            ),
            (
                "focus_loss_releases_held_buttons",
                focus_loss_releases_held_buttons,
            ),
        ];
        let mut failed = 0_usize;
        for (name, case) in cases {
            match case(&mut presenter) {
                Ok(()) => println!("test {name} ... ok"),
                Err(reason) => {
                    failed += 1;
                    println!("test {name} ... FAILED: {reason}");
                }
            }
        }
        match backing_density_keeps_the_window_and_fills_its_pixels(&mut presenter, &gateway) {
            Ok(()) => println!("test backing_density_keeps_the_window_and_fills_its_pixels ... ok"),
            Err(reason) => {
                failed += 1;
                println!(
                    "test backing_density_keeps_the_window_and_fills_its_pixels ... FAILED: {reason}"
                );
            }
        }
        let closing = closed_presenter_reports_closed(&mut presenter);
        match closing {
            Ok(()) => println!("test closed_presenter_reports_closed ... ok"),
            Err(reason) => {
                failed += 1;
                println!("test closed_presenter_reports_closed ... FAILED: {reason}");
            }
        }
        let total = cases.len() + 2;
        println!(
            "test result: {}. {} passed; {failed} failed",
            if failed == 0 { "ok" } else { "FAILED" },
            total - failed
        );
        if failed > 0 {
            std::process::exit(1);
        }
    }

    fn point(x: f64, y: f64) -> LogicalPoint {
        LogicalPoint { x, y }
    }

    fn post(presenter: &mut NativePresenter, input: SyntheticInput) -> Check {
        presenter
            .post_synthetic(&input)
            .map_err(|error| format!("post {input:?}: {error}"))
    }

    fn key(code: u16, text: &str, pressed: bool, modifiers: Modifiers) -> SyntheticInput {
        SyntheticInput::Key {
            code,
            text: text.into(),
            pressed,
            repeat: false,
            modifiers,
        }
    }

    /// Post a key-down marker, then poll until it arrives, and return every
    /// event before it. The marker bounds a case's events.
    fn until_marker(presenter: &mut NativePresenter, code: u16) -> Result<Vec<InputEvent>, String> {
        post(presenter, key(code, "", true, Modifiers::NONE))?;
        let mut events = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            presenter
                .poll_input(&mut events)
                .map_err(|error| format!("poll: {error}"))?;
            if let Some(index) = events.iter().position(|event| {
                matches!(event, InputEvent::Key { code: c, pressed: true, .. } if *c == code)
            }) {
                events.truncate(index);
                return Ok(events);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        Err(format!("marker key {code} never arrived; saw {events:?}"))
    }

    /// Poll until `found` matches an event, returning all events seen.
    fn until(
        presenter: &mut NativePresenter,
        within: Duration,
        found: impl Fn(&InputEvent) -> bool,
    ) -> Result<Vec<InputEvent>, String> {
        let mut events = Vec::new();
        let deadline = Instant::now() + within;
        while Instant::now() < deadline {
            presenter
                .poll_input(&mut events)
                .map_err(|error| format!("poll: {error}"))?;
            if events.iter().any(&found) {
                return Ok(events);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        Err(format!("expected event never arrived; saw {events:?}"))
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-9
    }

    fn same(a: LogicalPoint, b: LogicalPoint) -> bool {
        close(a.x, b.x) && close(a.y, b.y)
    }

    fn initial_focus_and_geometry(presenter: &mut NativePresenter) -> Check {
        let events = until(presenter, Duration::from_secs(5), |event| {
            matches!(event, InputEvent::FocusGained)
        })?;
        let metrics = events
            .iter()
            .find_map(|event| match event {
                InputEvent::Resized(metrics) => Some(*metrics),
                _ => None,
            })
            .ok_or_else(|| format!("no initial Resized in {events:?}"))?;
        if !(close(metrics.logical_width, f64::from(WIDTH))
            && close(metrics.logical_height, f64::from(HEIGHT))
            && metrics.drawable_width == WIDTH
            && metrics.drawable_height == HEIGHT
            && metrics.backing_scale >= 1.0)
        {
            return Err(format!("initial geometry {metrics:?}"));
        }
        let current = presenter
            .surface_metrics()
            .map_err(|error| format!("surface metrics: {error}"))?;
        if current != metrics {
            return Err(format!(
                "surface_metrics {current:?} differs from {metrics:?}"
            ));
        }
        if presenter
            .set_resizable(true)
            .map_err(|error| format!("set_resizable: {error}"))
            .is_err()
        {
            return Err("set_resizable failed".into());
        }
        Ok(())
    }

    fn pointer_motion_is_logical_top_left(presenter: &mut NativePresenter) -> Check {
        for position in [point(10.5, 20.25), point(11.0, 21.0), point(12.25, 22.75)] {
            post(
                presenter,
                SyntheticInput::PointerMoved {
                    position,
                    dragging: None,
                    modifiers: Modifiers::NONE,
                },
            )?;
        }
        let events = until_marker(presenter, KEY_SPACE)?;
        match events.last() {
            Some(InputEvent::PointerMoved {
                position,
                buttons,
                modifiers,
            }) if same(*position, point(12.25, 22.75))
                && buttons.is_empty()
                && *modifiers == Modifiers::NONE =>
            {
                Ok(())
            }
            other => Err(format!("last event before the marker was {other:?}")),
        }
    }

    fn primary_press_drag_release(presenter: &mut NativePresenter) -> Check {
        let primary = PointerButton::Primary;
        post(
            presenter,
            SyntheticInput::Button {
                button: primary,
                pressed: true,
                position: point(100.0, 50.0),
                clicks: 1,
                modifiers: Modifiers::NONE,
            },
        )?;
        post(
            presenter,
            SyntheticInput::PointerMoved {
                position: point(120.0, 60.5),
                dragging: Some(primary),
                modifiers: Modifiers::NONE,
            },
        )?;
        post(
            presenter,
            SyntheticInput::Button {
                button: primary,
                pressed: false,
                position: point(121.0, 61.0),
                clicks: 1,
                modifiers: Modifiers::NONE,
            },
        )?;
        let events = until_marker(presenter, KEY_SPACE)?;
        let ours: Vec<&InputEvent> = events
            .iter()
            .filter(|event| match event {
                InputEvent::Button { .. } => true,
                InputEvent::PointerMoved { buttons, .. } => !buttons.is_empty(),
                _ => false,
            })
            .collect();
        let held = ButtonSet::EMPTY.with(primary);
        let expected = [
            InputEvent::Button {
                button: primary,
                pressed: true,
                position: point(100.0, 50.0),
                clicks: 1,
                modifiers: Modifiers::NONE,
            },
            InputEvent::PointerMoved {
                position: point(120.0, 60.5),
                buttons: held,
                modifiers: Modifiers::NONE,
            },
            InputEvent::Button {
                button: primary,
                pressed: false,
                position: point(121.0, 61.0),
                clicks: 1,
                modifiers: Modifiers::NONE,
            },
        ];
        if ours.len() == expected.len() && ours.iter().zip(&expected).all(|(a, b)| *a == b) {
            Ok(())
        } else {
            Err(format!("press/drag/release gave {ours:?}"))
        }
    }

    fn secondary_click_carries_modifiers(presenter: &mut NativePresenter) -> Check {
        let command = Modifiers {
            command: true,
            ..Modifiers::NONE
        };
        for pressed in [true, false] {
            post(
                presenter,
                SyntheticInput::Button {
                    button: PointerButton::Secondary,
                    pressed,
                    position: point(30.0, 40.0),
                    clicks: 1,
                    modifiers: command,
                },
            )?;
        }
        let events = until_marker(presenter, KEY_SPACE)?;
        let buttons: Vec<(PointerButton, bool, bool)> = events
            .iter()
            .filter_map(|event| match event {
                InputEvent::Button {
                    button,
                    pressed,
                    modifiers,
                    ..
                } => Some((*button, *pressed, modifiers.command)),
                _ => None,
            })
            .collect();
        if buttons
            == [
                (PointerButton::Secondary, true, true),
                (PointerButton::Secondary, false, true),
            ]
        {
            Ok(())
        } else {
            Err(format!("secondary click gave {buttons:?}"))
        }
    }

    fn keys_carry_code_text_repeat_and_modifiers(presenter: &mut NativePresenter) -> Check {
        let shift = Modifiers {
            shift: true,
            ..Modifiers::NONE
        };
        post(presenter, key(KEY_A, "a", true, shift))?;
        post(
            presenter,
            SyntheticInput::Key {
                code: KEY_A,
                text: "a".into(),
                pressed: true,
                repeat: true,
                modifiers: shift,
            },
        )?;
        post(presenter, key(KEY_A, "a", false, shift))?;
        let events = until_marker(presenter, KEY_SPACE)?;
        let keys: Vec<&InputEvent> = events
            .iter()
            .filter(|event| matches!(event, InputEvent::Key { .. }))
            .collect();
        let expected = [
            InputEvent::Key {
                code: KEY_A,
                text: "a".into(),
                pressed: true,
                repeat: false,
                modifiers: shift,
            },
            InputEvent::Key {
                code: KEY_A,
                text: "a".into(),
                pressed: true,
                repeat: true,
                modifiers: shift,
            },
            InputEvent::Key {
                code: KEY_A,
                text: "a".into(),
                pressed: false,
                repeat: false,
                modifiers: shift,
            },
        ];
        if keys.len() == expected.len() && keys.iter().zip(&expected).all(|(a, b)| *a == b) {
            Ok(())
        } else {
            Err(format!("keys gave {keys:?}"))
        }
    }

    fn modifier_changes_are_typed(presenter: &mut NativePresenter) -> Check {
        let held = Modifiers {
            shift: true,
            option: true,
            ..Modifiers::NONE
        };
        post(
            presenter,
            SyntheticInput::ModifiersChanged { modifiers: held },
        )?;
        post(
            presenter,
            SyntheticInput::ModifiersChanged {
                modifiers: Modifiers::NONE,
            },
        )?;
        let events = until_marker(presenter, KEY_SPACE)?;
        let changes: Vec<Modifiers> = events
            .iter()
            .filter_map(|event| match event {
                InputEvent::ModifiersChanged { modifiers } => Some(*modifiers),
                _ => None,
            })
            .collect();
        if changes == [held, Modifiers::NONE] {
            Ok(())
        } else {
            Err(format!("modifier changes gave {changes:?}"))
        }
    }

    fn pixel_scroll_reaches_the_window(presenter: &mut NativePresenter) -> Check {
        post(
            presenter,
            SyntheticInput::Scroll {
                delta: [3, -7],
                position: point(50.0, 60.0),
            },
        )?;
        let events = until_marker(presenter, KEY_SPACE)?;
        let scrolls: Vec<&InputEvent> = events
            .iter()
            .filter(|event| matches!(event, InputEvent::Scroll { .. }))
            .collect();
        match scrolls.as_slice() {
            [
                InputEvent::Scroll {
                    delta,
                    unit,
                    position,
                    ..
                },
            ] if close(delta[0], 3.0)
                && close(delta[1], -7.0)
                && *unit == ScrollUnit::Points
                && (position.x - 50.0).abs() <= 1.0
                && (position.y - 60.0).abs() <= 1.0 =>
            {
                Ok(())
            }
            other => Err(format!("scroll gave {other:?}")),
        }
    }

    fn poll_events_keeps_input_queued(presenter: &mut NativePresenter) -> Check {
        post(presenter, key(KEY_A, "a", true, Modifiers::NONE))?;
        post(presenter, key(KEY_A, "a", false, Modifiers::NONE))?;
        for _ in 0..3 {
            presenter
                .poll_events()
                .map_err(|error| format!("poll_events: {error}"))?;
        }
        let events = until_marker(presenter, KEY_SPACE)?;
        let keys = events
            .iter()
            .filter(|event| matches!(event, InputEvent::Key { code: KEY_A, .. }))
            .count();
        if keys == 2 {
            Ok(())
        } else {
            Err(format!("poll_events lost queued keys: {events:?}"))
        }
    }

    fn resize_reports_new_geometry(presenter: &mut NativePresenter) -> Check {
        presenter
            .resize(400, 250)
            .map_err(|error| format!("resize: {error}"))?;
        let events = until(
            presenter,
            Duration::from_secs(2),
            |event| matches!(event, InputEvent::Resized(metrics) if metrics.drawable_width == 400),
        )?;
        let metrics = events
            .iter()
            .rev()
            .find_map(|event| match event {
                InputEvent::Resized(metrics) => Some(*metrics),
                _ => None,
            })
            .ok_or("no Resized")?;
        if close(metrics.logical_width, 400.0)
            && close(metrics.logical_height, 250.0)
            && metrics.drawable_height == 250
            && metrics.to_drawable(point(200.0, 125.0)) == Some([200.0, 125.0])
        {
            Ok(())
        } else {
            Err(format!("resized geometry {metrics:?}"))
        }
    }

    fn a_user_resize_leaves_the_drawable(presenter: &mut NativePresenter) -> Check {
        post(
            presenter,
            SyntheticInput::Resize {
                width: 360.0,
                height: 225.0,
            },
        )?;
        let events = until(
            presenter,
            Duration::from_secs(2),
            |event| matches!(event, InputEvent::Resized(metrics) if close(metrics.logical_width, 360.0)),
        )?;
        let metrics = events
            .iter()
            .rev()
            .find_map(|event| match event {
                InputEvent::Resized(metrics) => Some(*metrics),
                _ => None,
            })
            .ok_or("no Resized")?;
        // The drawable is still the 400 x 250 of the last presenter resize.
        if !(close(metrics.logical_height, 225.0)
            && metrics.drawable_width == 400
            && metrics.drawable_height == 250
            && metrics.to_drawable(point(180.0, 90.0)) == Some([200.0, 100.0]))
        {
            return Err(format!("user-resized geometry {metrics:?}"));
        }
        match presenter.post_synthetic(&SyntheticInput::Resize {
            width: 0.0,
            height: 10.0,
        }) {
            Err(PresentationError::Window(_)) => Ok(()),
            other => Err(format!("a zero-width resize gave {other:?}")),
        }
    }

    fn focus_loss_releases_held_buttons(presenter: &mut NativePresenter) -> Check {
        let primary = PointerButton::Primary;
        post(
            presenter,
            SyntheticInput::Button {
                button: primary,
                pressed: true,
                position: point(10.0, 10.0),
                clicks: 1,
                modifiers: Modifiers::NONE,
            },
        )?;
        let pressed = until_marker(presenter, KEY_ESCAPE)?;
        if !pressed
            .iter()
            .any(|event| matches!(event, InputEvent::Button { pressed: true, .. }))
        {
            return Err(format!("press never arrived: {pressed:?}"));
        }
        post(presenter, SyntheticInput::Focus(false))?;
        let lost = until(presenter, Duration::from_secs(5), |event| {
            matches!(event, InputEvent::FocusLost)
        })?;
        let tail: Vec<&InputEvent> = lost
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    InputEvent::FocusLost | InputEvent::PointerCaptureLost { .. }
                )
            })
            .collect();
        let expected = [
            InputEvent::PointerCaptureLost {
                released: ButtonSet::EMPTY.with(primary),
            },
            InputEvent::FocusLost,
        ];
        if !(tail.len() == 2 && tail.iter().zip(&expected).all(|(a, b)| *a == b)) {
            return Err(format!("focus loss gave {tail:?}"));
        }
        post(presenter, SyntheticInput::Focus(true))?;
        until(presenter, Duration::from_secs(5), |event| {
            matches!(event, InputEvent::FocusGained)
        })?;
        Ok(())
    }

    /// Pump every event posted so far, then read the geometry.
    fn settled(presenter: &mut NativePresenter) -> Result<SurfaceMetrics, String> {
        until_marker(presenter, KEY_SPACE)?;
        presenter
            .surface_metrics()
            .map_err(|error| format!("surface metrics: {error}"))
    }

    fn present(
        presenter: &mut NativePresenter,
        frame: &SharedBuffer,
        width: u32,
        height: u32,
    ) -> Check {
        presenter
            .present_rgba8(frame, width, height, width as usize * 4)
            .map(|_| ())
            .map_err(|error| format!("present {width} x {height}: {error}"))
    }

    fn contents_scale(presenter: &NativePresenter) -> Result<f64, String> {
        presenter
            .contents_scale()
            .map_err(|error| format!("contents scale: {error}"))
    }

    /// The content area, in points, and the drawable, in pixels.
    fn sizes(metrics: &SurfaceMetrics) -> ([f64; 2], [u32; 2]) {
        (
            [metrics.logical_width, metrics.logical_height],
            [metrics.drawable_width, metrics.drawable_height],
        )
    }

    fn backing_density_keeps_the_window_and_fills_its_pixels(
        presenter: &mut NativePresenter,
        gateway: &Gateway,
    ) -> Check {
        let before = settled(presenter)?;
        let scale = before.backing_scale;
        println!("native_input: backing scale {scale}");
        if presenter.pixel_density() != PixelDensity::OnePerPoint
            || contents_scale(presenter)? != 1.0
        {
            return Err(format!(
                "the default density is {:?} at contentsScale {}",
                presenter.pixel_density(),
                contents_scale(presenter)?
            ));
        }
        presenter
            .set_pixel_density(PixelDensity::Backing)
            .map_err(|error| format!("set_pixel_density: {error}"))?;
        if presenter.pixel_density() != PixelDensity::Backing || contents_scale(presenter)? != scale
        {
            return Err(format!(
                "backing density gave contentsScale {} on a {scale}x screen",
                contents_scale(presenter)?
            ));
        }
        let [width, height] = before
            .backing_size()
            .ok_or_else(|| format!("no backing size for {before:?}"))?;
        let frame = gateway
            .buffer_zeroed(width as usize * height as usize * 4)
            .map_err(|error| format!("frame buffer: {error}"))?;

        // A frame at the backing size fills the window without moving it.
        present(presenter, &frame, width, height)?;
        let shown = settled(presenter)?;
        let points = [before.logical_width, before.logical_height];
        if sizes(&shown) != (points, [width, height])
            || shown.to_drawable(point(180.0, 90.0)) != Some([180.0 * scale, 90.0 * scale])
            || contents_scale(presenter)? != scale
        {
            return Err(format!(
                "a {width} x {height} frame at backing density moved the window: {before:?} -> {shown:?}"
            ));
        }

        // A host resize moves the content area; the drawable waits for the
        // next frame.
        presenter
            .resize(300, 200)
            .map_err(|error| format!("resize: {error}"))?;
        let resized = settled(presenter)?;
        if sizes(&resized) != ([300.0, 200.0], [width, height]) {
            return Err(format!("a resize at backing density gave {resized:?}"));
        }
        let [width, height] = resized
            .backing_size()
            .ok_or_else(|| format!("no backing size for {resized:?}"))?;
        present(presenter, &frame, width, height)?;
        let followed = settled(presenter)?;
        if sizes(&followed) != ([300.0, 200.0], [width, height]) {
            return Err(format!("the next frame after the resize gave {followed:?}"));
        }

        // Back to one pixel per point: the window takes the drawable's size
        // in points, and each frame sizes it again.
        presenter
            .set_pixel_density(PixelDensity::OnePerPoint)
            .map_err(|error| format!("set_pixel_density: {error}"))?;
        let back = settled(presenter)?;
        if sizes(&back) != ([f64::from(width), f64::from(height)], [width, height])
            || contents_scale(presenter)? != 1.0
        {
            return Err(format!(
                "returning to one pixel per point gave {back:?} at contentsScale {}",
                contents_scale(presenter)?
            ));
        }
        present(presenter, &frame, 320, 200)?;
        let again = settled(presenter)?;
        if sizes(&again) != ([320.0, 200.0], [320, 200]) {
            return Err(format!(
                "a 320 x 200 frame at one pixel per point gave {again:?}"
            ));
        }
        Ok(())
    }

    fn closed_presenter_reports_closed(presenter: &mut NativePresenter) -> Check {
        presenter
            .close()
            .map_err(|error| format!("close: {error}"))?;
        let mut events = Vec::new();
        match presenter.poll_input(&mut events) {
            Ok(PresentationState::Closed) => {}
            other => return Err(format!("poll after close gave {other:?}")),
        }
        match presenter.post_synthetic(&SyntheticInput::Focus(true)) {
            Err(PresentationError::Closed) => Ok(()),
            other => Err(format!("post after close gave {other:?}")),
        }
    }
}
