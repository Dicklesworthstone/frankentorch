#![no_main]

use ft_dispatch::{
    DispatchKey, DispatchKeySet, ParsedSchemaInput, SchemaRegistry, parse_schema_name,
    parse_schema_or_name, schema_dispatch_key_from_tag, schema_dispatch_keyset_from_tags,
};
use libfuzzer_sys::fuzz_target;

const MAX_SCHEMA_INPUT_BYTES: usize = 8 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_SCHEMA_INPUT_BYTES {
        return;
    }

    let Ok(input) = std::str::from_utf8(data) else {
        return;
    };

    let name_result = parse_schema_name(input);
    let schema_result = parse_schema_or_name(input);

    if let Ok(ParsedSchemaInput::Name(parsed)) = &schema_result {
        assert_eq!(
            name_result.as_ref(),
            Ok(parsed),
            "name-only schema parsing must agree with direct name parsing"
        );
    }

    if let Ok(ParsedSchemaInput::Schema(schema)) = &schema_result {
        assert!(
            !schema.arguments.contains(") ->"),
            "arguments should be split before the return separator"
        );
        assert!(
            !schema.returns.is_empty(),
            "successful full schema parse must keep a non-empty return declaration"
        );
    }

    // Structure-aware path: the first line is a schema/name, remaining lines
    // are candidate dispatch key tags. This reaches SchemaRegistry::register
    // and duplicate-name handling, not only the text parser.
    let mut lines = input.lines();
    if let Some(schema_line) = lines.next() {
        let tags = lines
            .flat_map(|line| line.split([',', '|', ' ', '\n', '\t']))
            .filter(|tag| !tag.is_empty())
            .take(8)
            .collect::<Vec<_>>();
        if let Ok(parsed) = parse_schema_or_name(schema_line) {
            let keyset = if tags.is_empty() {
                Some(DispatchKeySet::from_keys(&[DispatchKey::CPU]))
            } else {
                schema_dispatch_keyset_from_tags(&tags).ok()
            };
            if let Some(keyset) = keyset {
                let mut registry = SchemaRegistry::new();
                if let Ok(normalized_name) = registry.register(&parsed, keyset) {
                    assert_eq!(registry.len(), 1);
                    assert!(!registry.is_empty());
                    let entry = registry
                        .lookup(normalized_name.as_str())
                        .expect("freshly registered schema must be lookupable");
                    assert_eq!(entry.normalized_name, normalized_name);
                    assert_eq!(entry.keyset.bits(), keyset.bits());
                    assert!(
                        registry.register(&parsed, keyset).is_err(),
                        "duplicate schema registration must be rejected"
                    );
                }
            }
        }
    }

    for tag in input
        .split([',', '|', ' ', '\n', '\t'])
        .filter(|tag| !tag.is_empty())
    {
        let _ = schema_dispatch_key_from_tag(tag);
    }

    let tags = input
        .split([',', '|', ' ', '\n', '\t'])
        .filter(|tag| !tag.is_empty())
        .take(8)
        .collect::<Vec<_>>();
    let _ = schema_dispatch_keyset_from_tags(&tags);
});
