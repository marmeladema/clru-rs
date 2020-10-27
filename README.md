# CLru

[![Actions](https://github.com/marmeladema/clru-rs/workflows/Rust/badge.svg)](https://github.com/marmeladema/clru-rs/actions)
[![Crate](https://img.shields.io/crates/v/clru)](https://crates.io/crates/clru)
[![Docs](https://docs.rs/clru/badge.svg)](https://docs.rs/clru)
[![License](https://img.shields.io/crates/l/clru)](LICENSE)

An [LRU cache](https://en.wikipedia.org/wiki/Cache_replacement_policies#Least_recently_used_(LRU)) implementation with constant time operations.

The cache is backed by a [HashMap](https://doc.rust-lang.org/std/collections/struct.HashMap.html) and thus offers a O(1) time complexity (amortized average) for common operations:
* `get` / `get_mut`
* `put` / `pop`
* `peek` / `peek_mut`

## Disclaimer

Most of the API, documentation, examples and tests have been heavily inspired by the [lru](https://github.com/jeromefroe/lru-rs) crate.
I want to thank [jeromefroe](https://github.com/jeromefroe/) for his work without which this crate would have probably never has been released.


## Differences with [lru](https://github.com/jeromefroe/lru-rs)

The main differences are:
* Smaller amount of unsafe code. Unsafe code is not bad in itself as long as it is thoroughly reviewed and understood but can be surprisingly hard to get right. Reducing the amount of unsafe code should hopefully reduce bugs or undefined behaviors.
* API closer to the standard [HashMap](https://doc.rust-lang.org/std/collections/struct.HashMap.html) collection which allows to lookup with `Borrow`-ed version of the key.

## Example

Below is a simple example of how to instantiate and use this LRU cache.

```rust
use clru::CLruCache;

fn main() {
    let mut cache = CLruCache::new(2);
    cache.put("apple".to_string(), 3);
    cache.put("banana".to_string(), 2);

    assert_eq!(cache.get("apple"), Some(&3));
    assert_eq!(cache.get("banana"), Some(&2));
    assert!(cache.get("pear").is_none());

    assert_eq!(cache.put("banana", 4), Some(2));
    assert_eq!(cache.put("pear", 5), None);

    assert_eq!(cache.get("pear"), Some(&5));
    assert_eq!(cache.get("banana"), Some(&4));
    assert!(cache.get("apple").is_none());

    {
        let v = cache.get_mut("banana").unwrap();
        *v = 6;
    }

    assert_eq!(cache.get("banana"), Some(&6));
}
```

## Tests

Each contribution is tested with regular compiler, miri, and 4 flavors of sanitizer (address, memory, thread and leak).
This should help catch bugs sooner than later.

## TODO

* improve documentation and add examples
* figure out `Send`/`Sync` traits support
