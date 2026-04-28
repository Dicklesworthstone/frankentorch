#![no_main]

use libfuzzer_sys::fuzz_target;

const MAX_NATIVE_STATE_DICT_BYTES: usize = 128 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_NATIVE_STATE_DICT_BYTES {
        return;
    }

    let _ = ft_serialize::load_state_dict_from_bytes(data);
});
