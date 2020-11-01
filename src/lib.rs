//! Another LRU cache implementation in rust.
//! The cache is backed by a [HashMap](https://doc.rust-lang.org/std/collections/struct.HashMap.html) and thus
//! offers a O(1) time complexity for common operations:
//! * `get` / `get_mut`
//! * `put` / `pop`
//! * `peek` / `peek_mut`
//!
//! ## Example
//!
//! ```rust
//!
//! use clru::CLruCache;
//!
//! let mut cache = CLruCache::new(2);
//! cache.put("apple".to_string(), 3);
//! cache.put("banana".to_string(), 2);
//!
//! assert_eq!(cache.get("apple"), Some(&3));
//! assert_eq!(cache.get("banana"), Some(&2));
//! assert!(cache.get("pear").is_none());
//!
//! assert_eq!(cache.put("banana".to_string(), 4), Some(2));
//! assert_eq!(cache.put("pear".to_string(), 5), None);
//!
//! assert_eq!(cache.get("pear"), Some(&5));
//! assert_eq!(cache.get("banana"), Some(&4));
//! assert!(cache.get("apple").is_none());
//!
//! {
//!     let v = cache.get_mut("banana").unwrap();
//!     *v = 6;
//! }
//!
//! assert_eq!(cache.get("banana"), Some(&6));
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]
#![deny(warnings)]

use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::rc::Rc;

#[derive(Debug)]
struct FixedSizeListNode<T> {
    prev: usize,
    next: usize,
    data: T,
}

#[derive(Debug)]
struct FixedSizeList<T> {
    nodes: Box<[Option<FixedSizeListNode<T>>]>,
    free: Vec<usize>,
    front: usize,
    back: usize,
}

impl<T> FixedSizeList<T> {
    fn new(capacity: usize) -> Self {
        Self {
            nodes: {
                let mut vec = Vec::with_capacity(capacity);
                vec.resize_with(capacity, Default::default);
                vec.into_boxed_slice()
            },
            free: (0..capacity).rev().collect(),
            front: usize::MAX,
            back: usize::MAX,
        }
    }

    fn capacity(&self) -> usize {
        self.nodes.len()
    }

    fn len(&self) -> usize {
        self.nodes.len() - self.free.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn is_full(&self) -> bool {
        self.len() == self.capacity()
    }

    fn clear(&mut self) {
        self.nodes.iter_mut().for_each(|node| *node = None);
        self.free.clear();
        for i in 0..self.capacity() {
            self.free.push(i);
        }
        self.front = usize::MAX;
        self.back = usize::MAX;
    }

    fn node_ref(&self, idx: usize) -> Option<&FixedSizeListNode<T>> {
        self.nodes.get(idx).and_then(|node| node.as_ref())
    }

    fn node_mut(&mut self, idx: usize) -> Option<&mut FixedSizeListNode<T>> {
        self.nodes.get_mut(idx).and_then(|node| node.as_mut())
    }

    fn front(&self) -> Option<&T> {
        self.node_ref(self.front).map(|node| &node.data)
    }

    fn front_mut(&mut self) -> Option<&mut T> {
        self.node_mut(self.front).map(|node| &mut node.data)
    }

    fn back(&self) -> Option<&T> {
        self.node_ref(self.back).map(|node| &node.data)
    }

    fn back_mut(&mut self) -> Option<&mut T> {
        self.node_mut(self.back).map(|node| &mut node.data)
    }

    fn push_front(&mut self, data: T) -> Option<(usize, &mut T)> {
        let idx = self.free.pop()?;
        self.nodes[idx] = Some(FixedSizeListNode {
            prev: usize::MAX,
            next: self.front,
            data,
        });
        if let Some(front) = self.node_mut(self.front) {
            front.prev = idx;
        }
        if self.node_ref(self.back).is_none() {
            self.back = idx;
        }
        self.front = idx;
        Some((idx, &mut self.nodes[idx].as_mut().unwrap().data))
    }

    #[cfg(test)]
    fn push_back(&mut self, data: T) -> Option<(usize, &mut T)> {
        let idx = self.free.pop()?;
        self.nodes[idx] = Some(FixedSizeListNode {
            prev: self.back,
            next: usize::MAX,
            data,
        });
        if let Some(back) = self.node_mut(self.back) {
            back.next = idx;
        }
        if self.node_ref(self.front).is_none() {
            self.front = idx;
        }
        self.back = idx;
        Some((idx, &mut self.nodes[idx].as_mut().unwrap().data))
    }

    fn pop_front(&mut self) -> Option<T> {
        self.remove(self.front)
    }

    fn pop_back(&mut self) -> Option<T> {
        self.remove(self.back)
    }

    fn remove(&mut self, idx: usize) -> Option<T> {
        let node = self.nodes.get_mut(idx)?.take()?;
        if let Some(prev) = self.node_mut(node.prev) {
            prev.next = node.next;
        } else {
            self.front = node.next;
        }
        if let Some(next) = self.node_mut(node.next) {
            next.prev = node.prev;
        } else {
            self.back = node.prev;
        }
        self.free.push(idx);
        Some(node.data)
    }

    fn iter(&self) -> FixedSizeListIter<'_, T> {
        FixedSizeListIter {
            list: self,
            front: self.front,
            back: self.back,
            len: self.len(),
        }
    }

    fn iter_mut(&mut self) -> FixedSizeListIterMut<'_, T> {
        let front = self.front;
        let back = self.back;
        FixedSizeListIterMut {
            ptr: self,
            front,
            back,
            len: self.len(),
            _marker: PhantomData,
        }
    }

    fn reorder(&mut self) {
        if self.is_empty() {
            return;
        }

        let len = self.len();
        let mut current = 0;
        while current < len {
            let front = self.front;
            let front_data = self.pop_front().unwrap();
            if front != current {
                debug_assert!(current < front, "{} < {}", current, front);
                // We need to free self.nodes[current] if its occupied
                if let Some(current_node) = self.nodes[current].take() {
                    if let Some(node) = self.node_mut(current_node.prev) {
                        node.next = front;
                    } else {
                        self.front = front;
                    }
                    if let Some(node) = self.node_mut(current_node.next) {
                        node.prev = front;
                    } else {
                        self.back = front;
                    }
                    self.nodes[front] = Some(current_node);
                }
            }
            // Assign new front node
            self.nodes[current] = Some(FixedSizeListNode {
                prev: current.wrapping_sub(1),
                next: current + 1,
                data: front_data,
            });
            current += 1;
        }
        self.front = 0;
        self.nodes[len - 1].as_mut().unwrap().next = usize::MAX;
        self.back = len - 1;
        self.free.clear();
        self.free.extend((len..self.capacity()).rev());
    }

    fn resize(&mut self, capacity: usize) {
        let len = self.len();
        let cap = self.capacity();
        match capacity.cmp(&cap) {
            Ordering::Less => {
                self.reorder();
                let mut nodes = std::mem::take(&mut self.nodes).into_vec();
                nodes.truncate(capacity);
                self.nodes = nodes.into_boxed_slice();
                self.free.clear();
                self.free.extend((len..self.nodes.len()).rev());
            }
            Ordering::Equal => {}
            Ordering::Greater => {
                let mut nodes = std::mem::take(&mut self.nodes).into_vec();
                nodes.extend((cap..capacity).map(|_| None));
                self.nodes = nodes.into_boxed_slice();
                self.free.extend((cap..self.nodes.len()).rev());
            }
        };
        debug_assert_eq!(self.len(), len);
        debug_assert_eq!(self.capacity(), capacity);
    }
}

#[derive(Clone, Debug)]
struct FixedSizeListIter<'a, T> {
    list: &'a FixedSizeList<T>,
    front: usize,
    back: usize,
    len: usize,
}

impl<'a, T> Iterator for FixedSizeListIter<'a, T> {
    type Item = (usize, &'a T);

    fn next(&mut self) -> Option<Self::Item> {
        if self.len > 0 {
            let front = self.front;
            let node = self.list.node_ref(front).unwrap();
            self.front = node.next;
            self.len -= 1;
            Some((front, &node.data))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, T> DoubleEndedIterator for FixedSizeListIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.len > 0 {
            let back = self.back;
            let node = self.list.node_ref(back).unwrap();
            self.back = node.prev;
            self.len -= 1;
            Some((back, &node.data))
        } else {
            None
        }
    }
}

impl<'a, T> ExactSizeIterator for FixedSizeListIter<'a, T> {
    fn len(&self) -> usize {
        self.size_hint().0
    }
}

struct FixedSizeListIterMut<'a, T> {
    ptr: *mut FixedSizeList<T>,
    front: usize,
    back: usize,
    len: usize,
    _marker: PhantomData<&'a mut FixedSizeList<T>>,
}

impl<'a, T> Iterator for FixedSizeListIterMut<'a, T> {
    type Item = (usize, &'a mut T);

    #[allow(unsafe_code)]
    fn next(&mut self) -> Option<Self::Item> {
        // Safety: self.ptr has been created from a valid mutable reference
        let list: &'a mut FixedSizeList<T> = unsafe { &mut *self.ptr };
        if self.len > 0 {
            let front = self.front;
            let node = list.node_mut(front).unwrap();
            self.front = node.next;
            self.len -= 1;
            Some((front, &mut node.data))
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, T> DoubleEndedIterator for FixedSizeListIterMut<'a, T> {
    #[allow(unsafe_code)]
    fn next_back(&mut self) -> Option<Self::Item> {
        // Safety: self.ptr has been created from a valid mutable reference
        let list: &'a mut FixedSizeList<T> = unsafe { &mut *self.ptr };
        if self.len > 0 {
            let back = self.back;
            let node = list.node_mut(back).unwrap();
            self.back = node.prev;
            self.len -= 1;
            Some((back, &mut node.data))
        } else {
            None
        }
    }
}

impl<'a, T> ExactSizeIterator for FixedSizeListIterMut<'a, T> {
    fn len(&self) -> usize {
        self.size_hint().0
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct Key<K>(Rc<K>);

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
#[repr(transparent)]
struct KeyRef<Q: ?Sized>(Q);

impl<Q: ?Sized> From<&Q> for &KeyRef<Q> {
    #[allow(unsafe_code)]
    fn from(value: &Q) -> Self {
        // Safety: this is safe because `KeyRef` is a newtype around Q
        // and is marked as `#[repr(transparent)]`
        unsafe { &*(value as *const Q as *const KeyRef<Q>) }
    }
}

impl<Q: ?Sized, K: Borrow<Q>> Borrow<KeyRef<Q>> for Key<K> {
    fn borrow(&self) -> &KeyRef<Q> {
        (&*self.0).borrow().into()
    }
}

/// An LRU cache with constant time operations.
pub struct CLruCache<K, V, S = RandomState> {
    lookup: HashMap<Key<K>, usize, S>,
    storage: FixedSizeList<(Key<K>, V)>,
}

impl<K: Eq + Hash, V> CLruCache<K, V> {
    /// Creates a new LRU Cache that holds at most `capacity` items.
    pub fn new(capacity: usize) -> Self {
        Self {
            lookup: HashMap::with_capacity(capacity),
            storage: FixedSizeList::new(capacity),
        }
    }
}

impl<K: Eq + Hash, V, S: BuildHasher> CLruCache<K, V, S> {
    /// Creates a new LRU Cache that holds at most `capacity` items and uses the provided hash builder to hash keys.
    pub fn with_hasher(capacity: usize, hash_builder: S) -> Self {
        Self {
            lookup: HashMap::with_capacity_and_hasher(capacity, hash_builder),
            storage: FixedSizeList::new(capacity),
        }
    }

    /// Returns the number of key-value pairs that are currently in the the cache.
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.lookup.len(), self.storage.len());
        self.storage.len()
    }

    /// Returns the maximum number of key-value pairs the cache can hold.
    pub fn capacity(&self) -> usize {
        self.storage.capacity()
    }

    /// Returns a bool indicating whether the cache is empty or not.
    pub fn is_empty(&self) -> bool {
        debug_assert_eq!(self.lookup.is_empty(), self.storage.is_empty());
        self.storage.is_empty()
    }

    /// Returns a bool indicating whether the cache is full or not.
    pub fn is_full(&self) -> bool {
        debug_assert_eq!(
            self.lookup.len() == self.lookup.capacity(),
            self.storage.is_full()
        );
        self.storage.is_full()
    }

    /// Returns the value corresponding to the most recently used item or `None` if the cache is empty.
    /// Like `peek`, `font` does not update the LRU list so the item's position will be unchanged.
    pub fn front(&self) -> Option<(&K, &V)> {
        self.storage.front().map(|(key, data)| (&*key.0, data))
    }

    /// Returns the value corresponding to the most recently used item or `None` if the cache is empty.
    /// Like `peek`, `font` does not update the LRU list so the item's position will be unchanged.
    pub fn front_mut(&mut self) -> Option<(&K, &mut V)> {
        self.storage.front_mut().map(|(key, data)| (&*key.0, data))
    }

    /// Returns the value corresponding to the least recently used item or `None` if the cache is empty.
    /// Like `peek`, `back` does not update the LRU list so the item's position will be unchanged.
    pub fn back(&self) -> Option<(&K, &V)> {
        self.storage.back().map(|(key, value)| (&*key.0, value))
    }

    /// Returns the value corresponding to the least recently used item or `None` if the cache is empty.
    /// Like `peek`, `back` does not update the LRU list so the item's position will be unchanged.
    pub fn back_mut(&mut self) -> Option<(&K, &mut V)> {
        self.storage.back_mut().map(|(key, value)| (&*key.0, value))
    }

    /// Puts a key-value pair into cache.
    /// If the key already exists in the cache, then it updates the key's value and returns the old value.
    /// Otherwise, `None` is returned.
    pub fn put(&mut self, key: K, value: V) -> Option<V> {
        match self.lookup.entry(Key(Rc::new(key))) {
            Entry::Occupied(mut occ) => {
                let old = self.storage.remove(*occ.get());
                let (idx, _) = self
                    .storage
                    .push_front((Key(occ.key().0.clone()), value))
                    .unwrap();
                occ.insert(idx);
                old.map(|(_, data)| data)
            }
            Entry::Vacant(vac) => {
                let mut obsolete_key = None;
                if self.storage.is_full() {
                    obsolete_key = self.storage.pop_back().map(|(key, _)| key);
                }
                if let Some((idx, _)) = self.storage.push_front((Key(vac.key().0.clone()), value)) {
                    vac.insert(idx);
                }
                if let Some(obsolete_key) = obsolete_key {
                    self.lookup.remove(&obsolete_key);
                }
                None
            }
        }
    }

    /// Returns a reference to the value of the key in the cache or `None` if it is not present in the cache.
    /// Moves the key to the head of the LRU list if it exists.
    pub fn get<Q: ?Sized>(&mut self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let key: &KeyRef<Q> = key.into();
        let idx = *self.lookup.get(key)?;
        let value = self.storage.remove(idx)?;
        self.storage.push_front(value).map(|(_, (_, data))| &*data)
    }

    /// Returns a mutable reference to the value of the key in the cache or `None` if it is not present in the cache.
    /// Moves the key to the head of the LRU list if it exists.
    pub fn get_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let key: &KeyRef<Q> = key.into();
        let idx = *self.lookup.get(key)?;
        let value = self.storage.remove(idx)?;
        self.storage.push_front(value).map(|(_, (_, data))| data)
    }

    /// Removes and returns the value corresponding to the key from the cache or `None` if it does not exist.
    pub fn pop<Q: ?Sized>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let key: &KeyRef<Q> = key.into();
        let idx = self.lookup.remove(key)?;
        self.storage.remove(idx).map(|(_, value)| value)
    }

    /// Returns a reference to the value corresponding to the key in the cache or `None` if it is not present in the cache.
    /// Unlike `get`, `peek` does not update the LRU list so the key's position will be unchanged.
    pub fn peek<Q: ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let key: &KeyRef<Q> = key.into();
        let idx = *self.lookup.get(key)?;
        self.storage.node_ref(idx).map(|node| &node.data.1)
    }

    /// Returns a mutable reference to the value corresponding to the key in the cache or `None` if it is not present in the cache.
    /// Unlike `get_mut`, `peek_mut` does not update the LRU list so the key's position will be unchanged.
    pub fn peek_mut<Q: ?Sized>(&mut self, key: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let key: &KeyRef<Q> = key.into();
        let idx = *self.lookup.get(key)?;
        self.storage.node_mut(idx).map(|node| &mut node.data.1)
    }

    /// Returns a bool indicating whether the given key is in the cache.
    /// Does not update the LRU list.
    pub fn contains<Q: ?Sized>(&mut self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.peek(key).is_some()
    }

    /// Clears the contents of the cache.
    pub fn clear(&mut self) {
        self.lookup.clear();
        self.storage.clear();
    }

    /// Returns an iterator visiting all entries in order.
    /// The iterator element type is `(&'a K, &'a V)`.
    pub fn iter(&self) -> CLruCacheIter<'_, K, V> {
        CLruCacheIter {
            iter: self.storage.iter(),
        }
    }

    /// Returns an iterator visiting all entries in order, giving a mutable reference on V.
    /// The iterator element type is `(&'a K, &'a mut V)`.
    pub fn iter_mut(&mut self) -> CLruCacheIterMut<'_, K, V> {
        CLruCacheIterMut {
            iter: self.storage.iter_mut(),
        }
    }

    /// Resizes the cache.
    /// If the new capacity is smaller than the size of the current cache any entries past the new capacity are discarded.
    pub fn resize(&mut self, capacity: usize) {
        while capacity < self.storage.len() {
            if let Some((key, _)) = self.storage.pop_back() {
                self.lookup.remove(&key).unwrap();
            }
        }
        self.storage.resize(capacity);
        for i in 0..self.len() {
            let FixedSizeListNode { data, .. } = self.storage.node_ref(i).unwrap();
            *self.lookup.get_mut(&data.0).unwrap() = i;
        }
    }
}

/// An iterator over the entries of a `CLruCache`.
///
/// This `struct` is created by the [`iter`] method on [`CLruCache`][`CLruCache`].
/// See its documentation for more.
///
/// [`iter`]: struct.CLruCache.html#method.iter
/// [`CLruCache`]: struct.CLruCache.html
#[derive(Clone, Debug)]
pub struct CLruCacheIter<'a, K, V> {
    iter: FixedSizeListIter<'a, (Key<K>, V)>,
}

impl<'a, K, V> Iterator for CLruCacheIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, (k, v))| (k.0.borrow(), v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K, V> DoubleEndedIterator for CLruCacheIter<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|(_, (k, v))| (k.0.borrow(), v))
    }
}

impl<'a, K, V> ExactSizeIterator for CLruCacheIter<'a, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

/// An iterator over mutables entries of a `CLruCache`.
///
/// This `struct` is created by the [`iter_mut`] method on [`CLruCache`][`CLruCache`].
/// See its documentation for more.
///
/// [`iter_mut`]: struct.CLruCache.html#method.iter_mut
/// [`CLruCache`]: struct.CLruCache.html
pub struct CLruCacheIterMut<'a, K, V> {
    iter: FixedSizeListIterMut<'a, (Key<K>, V)>,
}

impl<'a, K, V> Iterator for CLruCacheIterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|(_, (k, v))| ((*k).0.borrow(), v))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K, V> DoubleEndedIterator for CLruCacheIterMut<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter
            .next_back()
            .map(|(_, (k, v))| ((*k).0.borrow(), v))
    }
}

impl<'a, K, V> ExactSizeIterator for CLruCacheIterMut<'a, K, V> {
    fn len(&self) -> usize {
        self.iter.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_size_list() {
        let mut list = FixedSizeList::new(4);

        assert!(list.is_empty());
        assert_eq!(list.len(), 0);

        assert_eq!(list.front(), None);
        assert_eq!(list.front_mut(), None);

        assert_eq!(list.back(), None);
        assert_eq!(list.back_mut(), None);

        assert_eq!(list.iter().count(), 0);
        assert_eq!(list.iter().rev().count(), 0);

        assert_eq!(list.push_front(7), Some((0, &mut 7)));

        assert!(!list.is_empty());
        assert_eq!(list.len(), 1);

        assert_eq!(list.front(), Some(&7));
        assert_eq!(list.front_mut(), Some(&mut 7));

        assert_eq!(list.back(), Some(&7));
        assert_eq!(list.back_mut(), Some(&mut 7));

        assert_eq!(list.iter().collect::<Vec<_>>(), vec![(0, &7)]);
        assert_eq!(list.iter().rev().collect::<Vec<_>>(), vec![(0, &7)]);

        assert_eq!(list.push_front(5), Some((1, &mut 5)));

        assert!(!list.is_empty());
        assert_eq!(list.len(), 2);

        assert_eq!(list.front(), Some(&5));
        assert_eq!(list.front_mut(), Some(&mut 5));

        assert_eq!(list.back(), Some(&7));
        assert_eq!(list.back_mut(), Some(&mut 7));

        assert_eq!(list.iter().collect::<Vec<_>>(), vec![(1, &5), (0, &7)]);
        assert_eq!(
            list.iter().rev().collect::<Vec<_>>(),
            vec![(0, &7), (1, &5)]
        );

        assert_eq!(list.push_front(3), Some((2, &mut 3)));

        assert!(!list.is_empty());
        assert_eq!(list.len(), 3);

        assert_eq!(list.front(), Some(&3));
        assert_eq!(list.front_mut(), Some(&mut 3));

        assert_eq!(list.back(), Some(&7));
        assert_eq!(list.back_mut(), Some(&mut 7));

        assert_eq!(
            list.iter().collect::<Vec<_>>(),
            vec![(2, &3), (1, &5), (0, &7)]
        );
        assert_eq!(
            list.iter().rev().collect::<Vec<_>>(),
            vec![(0, &7), (1, &5), (2, &3)]
        );

        list.remove(1);

        assert!(!list.is_empty());
        assert_eq!(list.len(), 2);

        assert_eq!(list.front(), Some(&3));
        assert_eq!(list.front_mut(), Some(&mut 3));

        assert_eq!(list.back(), Some(&7));
        assert_eq!(list.back_mut(), Some(&mut 7));

        assert_eq!(list.iter().collect::<Vec<_>>(), vec![(2, &3), (0, &7)]);
        assert_eq!(
            list.iter().rev().collect::<Vec<_>>(),
            vec![(0, &7), (2, &3)]
        );

        list.remove(0);

        assert!(!list.is_empty());
        assert_eq!(list.len(), 1);

        assert_eq!(list.front(), Some(&3));
        assert_eq!(list.front_mut(), Some(&mut 3));

        assert_eq!(list.back(), Some(&3));
        assert_eq!(list.back_mut(), Some(&mut 3));

        assert_eq!(list.iter().collect::<Vec<_>>(), vec![(2, &3)]);
        assert_eq!(list.iter().rev().collect::<Vec<_>>(), vec![(2, &3)]);

        list.remove(2);

        assert!(list.is_empty());
        assert_eq!(list.len(), 0);

        assert_eq!(list.front(), None);
        assert_eq!(list.front_mut(), None);

        assert_eq!(list.back(), None);
        assert_eq!(list.back_mut(), None);

        assert_eq!(list.iter().count(), 0);
        assert_eq!(list.iter().rev().count(), 0);
    }

    #[test]
    fn test_fixed_size_list_reorder() {
        let mut list = FixedSizeList::new(4);

        list.push_back('a');
        list.push_front('b');
        list.push_back('c');
        list.push_front('d');

        assert_eq!(
            list.iter().collect::<Vec<_>>(),
            vec![(3, &'d'), (1, &'b'), (0, &'a'), (2, &'c')]
        );

        list.reorder();

        assert_eq!(
            list.iter().collect::<Vec<_>>(),
            vec![(0, &'d'), (1, &'b'), (2, &'a'), (3, &'c')]
        );
    }

    #[test]
    fn test_insert_and_get() {
        let mut cache = CLruCache::new(2);
        assert!(cache.is_empty());

        assert_eq!(cache.put("apple", "red"), None);
        assert_eq!(cache.put("banana", "yellow"), None);

        assert_eq!(cache.capacity(), 2);
        assert_eq!(cache.len(), 2);
        assert!(!cache.is_empty());
        assert_eq!(cache.get(&"apple"), Some(&"red"));
        assert_eq!(cache.get(&"banana"), Some(&"yellow"));
    }

    #[test]
    fn test_insert_and_get_mut() {
        let mut cache = CLruCache::new(2);

        cache.put("apple", "red");
        cache.put("banana", "yellow");

        assert_eq!(cache.capacity(), 2);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get_mut(&"apple"), Some(&mut "red"));
        assert_eq!(cache.get_mut(&"banana"), Some(&mut "yellow"));
    }

    #[test]
    fn test_get_mut_and_update() {
        let mut cache = CLruCache::new(2);

        cache.put("apple", 1);
        cache.put("banana", 3);

        {
            let v = cache.get_mut(&"apple").unwrap();
            *v = 4;
        }

        assert_eq!(cache.capacity(), 2);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get_mut(&"apple"), Some(&mut 4));
        assert_eq!(cache.get_mut(&"banana"), Some(&mut 3));
    }

    #[test]
    fn test_insert_update() {
        let mut cache = CLruCache::new(1);

        assert_eq!(cache.put("apple", "red"), None);
        assert_eq!(cache.put("apple", "green"), Some("red"));

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&"apple"), Some(&"green"));
    }

    #[test]
    fn test_insert_removes_oldest() {
        let mut cache = CLruCache::new(2);

        assert_eq!(cache.put("apple", "red"), None);
        assert_eq!(cache.put("banana", "yellow"), None);
        assert_eq!(cache.put("pear", "green"), None);

        assert!(cache.get(&"apple").is_none());
        assert_eq!(cache.get(&"banana"), Some(&"yellow"));
        assert_eq!(cache.get(&"pear"), Some(&"green"));

        // Even though we inserted "apple" into the cache earlier it has since been removed from
        // the cache so there is no current value for `insert` to return.
        assert_eq!(cache.put("apple", "green"), None);
        assert_eq!(cache.put("tomato", "red"), None);

        assert!(cache.get(&"pear").is_none());
        assert_eq!(cache.get(&"apple"), Some(&"green"));
        assert_eq!(cache.get(&"tomato"), Some(&"red"));
    }

    #[test]
    fn test_peek() {
        let mut cache = CLruCache::new(2);

        cache.put("apple", "red");
        cache.put("banana", "yellow");

        assert_eq!(cache.peek(&"banana"), Some(&"yellow"));
        assert_eq!(cache.peek(&"apple"), Some(&"red"));

        cache.put("pear", "green");

        assert!(cache.peek(&"apple").is_none());
        assert_eq!(cache.peek(&"banana"), Some(&"yellow"));
        assert_eq!(cache.peek(&"pear"), Some(&"green"));
    }

    #[test]
    fn test_peek_mut() {
        let mut cache = CLruCache::new(2);

        cache.put("apple", "red");
        cache.put("banana", "yellow");

        assert_eq!(cache.peek_mut(&"banana"), Some(&mut "yellow"));
        assert_eq!(cache.peek_mut(&"apple"), Some(&mut "red"));
        assert!(cache.peek_mut(&"pear").is_none());

        cache.put("pear", "green");

        assert!(cache.peek_mut(&"apple").is_none());
        assert_eq!(cache.peek_mut(&"banana"), Some(&mut "yellow"));
        assert_eq!(cache.peek_mut(&"pear"), Some(&mut "green"));

        {
            let v = cache.peek_mut(&"banana").unwrap();
            *v = "green";
        }

        assert_eq!(cache.peek_mut(&"banana"), Some(&mut "green"));
    }

    #[test]
    fn test_contains() {
        let mut cache = CLruCache::new(2);

        cache.put("apple", "red");
        cache.put("banana", "yellow");
        cache.put("pear", "green");

        assert!(!cache.contains(&"apple"));
        assert!(cache.contains(&"banana"));
        assert!(cache.contains(&"pear"));
    }

    #[test]
    fn test_pop() {
        let mut cache = CLruCache::new(2);

        cache.put("apple", "red");
        cache.put("banana", "yellow");

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&"apple"), Some(&"red"));
        assert_eq!(cache.get(&"banana"), Some(&"yellow"));

        let popped = cache.pop(&"apple");
        assert!(popped.is_some());
        assert_eq!(popped.unwrap(), "red");
        assert_eq!(cache.len(), 1);
        assert!(cache.get(&"apple").is_none());
        assert_eq!(cache.get(&"banana"), Some(&"yellow"));
    }

    #[test]
    fn test_clear() {
        let mut cache = CLruCache::new(2);

        cache.put("apple", "red");
        cache.put("banana", "yellow");

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&"apple"), Some(&"red"));
        assert_eq!(cache.get(&"banana"), Some(&"yellow"));

        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_resize_larger() {
        let mut cache = CLruCache::new(2);

        cache.put(1, "a");
        cache.put(2, "b");

        cache.resize(3);

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.capacity(), 3);

        cache.resize(4);

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.capacity(), 4);

        cache.put(3, "c");
        cache.put(4, "d");

        assert_eq!(cache.len(), 4);
        assert_eq!(cache.capacity(), 4);
        assert_eq!(cache.get(&1), Some(&"a"));
        assert_eq!(cache.get(&2), Some(&"b"));
        assert_eq!(cache.get(&3), Some(&"c"));
        assert_eq!(cache.get(&4), Some(&"d"));
    }

    #[test]
    fn test_resize_smaller() {
        let mut cache = CLruCache::new(4);

        cache.put(1, "a");
        cache.put(2, "b");
        cache.put(3, "c");
        cache.put(4, "d");

        cache.resize(2);

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.capacity(), 2);
        assert!(cache.get(&1).is_none());
        assert!(cache.get(&2).is_none());
        assert_eq!(cache.get(&3), Some(&"c"));
        assert_eq!(cache.get(&4), Some(&"d"));
    }

    #[test]
    fn test_resize_equal() {
        let mut cache = CLruCache::new(4);

        cache.put(1, "a");
        cache.put(2, "b");
        cache.put(3, "c");
        cache.put(4, "d");

        cache.resize(4);

        assert_eq!(cache.len(), 4);
        assert_eq!(cache.capacity(), 4);
        assert_eq!(cache.get(&1), Some(&"a"));
        assert_eq!(cache.get(&2), Some(&"b"));
        assert_eq!(cache.get(&3), Some(&"c"));
        assert_eq!(cache.get(&4), Some(&"d"));
    }

    #[test]
    fn test_iter_forwards() {
        let mut cache = CLruCache::new(3);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);

        {
            // iter const
            let mut iter = cache.iter();
            assert_eq!(iter.len(), 3);
            assert_eq!(iter.next(), Some((&"c", &3)));

            assert_eq!(iter.len(), 2);
            assert_eq!(iter.next(), Some((&"b", &2)));

            assert_eq!(iter.len(), 1);
            assert_eq!(iter.next(), Some((&"a", &1)));

            assert_eq!(iter.len(), 0);
            assert_eq!(iter.next(), None);
        }
        {
            // iter mut
            let mut iter = cache.iter_mut();
            assert_eq!(iter.len(), 3);
            assert_eq!(iter.next(), Some((&"c", &mut 3)));

            assert_eq!(iter.len(), 2);
            assert_eq!(iter.next(), Some((&"b", &mut 2)));

            assert_eq!(iter.len(), 1);
            assert_eq!(iter.next(), Some((&"a", &mut 1)));

            assert_eq!(iter.len(), 0);
            assert_eq!(iter.next(), None);
        }
    }

    #[test]
    fn test_iter_backwards() {
        let mut cache = CLruCache::new(3);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);

        {
            // iter const
            let mut iter = cache.iter();
            assert_eq!(iter.len(), 3);
            assert_eq!(iter.next_back(), Some((&"a", &1)));

            assert_eq!(iter.len(), 2);
            assert_eq!(iter.next_back(), Some((&"b", &2)));

            assert_eq!(iter.len(), 1);
            assert_eq!(iter.next_back(), Some((&"c", &3)));

            assert_eq!(iter.len(), 0);
            assert_eq!(iter.next_back(), None);
        }

        {
            // iter mut
            let mut iter = cache.iter_mut();
            assert_eq!(iter.len(), 3);
            assert_eq!(iter.next_back(), Some((&"a", &mut 1)));

            assert_eq!(iter.len(), 2);
            assert_eq!(iter.next_back(), Some((&"b", &mut 2)));

            assert_eq!(iter.len(), 1);
            assert_eq!(iter.next_back(), Some((&"c", &mut 3)));

            assert_eq!(iter.len(), 0);
            assert_eq!(iter.next_back(), None);
        }
    }

    #[test]
    fn test_iter_forwards_and_backwards() {
        let mut cache = CLruCache::new(3);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);

        {
            // iter const
            let mut iter = cache.iter();
            assert_eq!(iter.len(), 3);
            assert_eq!(iter.next(), Some((&"c", &3)));

            assert_eq!(iter.len(), 2);
            assert_eq!(iter.next_back(), Some((&"a", &1)));

            assert_eq!(iter.len(), 1);
            assert_eq!(iter.next(), Some((&"b", &2)));

            assert_eq!(iter.len(), 0);
            assert_eq!(iter.next_back(), None);
        }
        {
            // iter mut
            let mut iter = cache.iter_mut();
            assert_eq!(iter.len(), 3);
            assert_eq!(iter.next(), Some((&"c", &mut 3)));

            assert_eq!(iter.len(), 2);
            assert_eq!(iter.next_back(), Some((&"a", &mut 1)));

            assert_eq!(iter.len(), 1);
            assert_eq!(iter.next(), Some((&"b", &mut 2)));

            assert_eq!(iter.len(), 0);
            assert_eq!(iter.next_back(), None);
        }
    }

    #[test]
    fn test_iter_clone() {
        let mut cache = CLruCache::new(3);
        cache.put("a", 1);
        cache.put("b", 2);

        let mut iter = cache.iter();
        let mut iter_clone = iter.clone();

        assert_eq!(iter.len(), 2);
        assert_eq!(iter.next(), Some((&"b", &2)));
        assert_eq!(iter_clone.len(), 2);
        assert_eq!(iter_clone.next(), Some((&"b", &2)));

        assert_eq!(iter.len(), 1);
        assert_eq!(iter.next(), Some((&"a", &1)));
        assert_eq!(iter_clone.len(), 1);
        assert_eq!(iter_clone.next(), Some((&"a", &1)));

        assert_eq!(iter.len(), 0);
        assert_eq!(iter.next(), None);
        assert_eq!(iter_clone.len(), 0);
        assert_eq!(iter_clone.next(), None);
    }

    #[test]
    fn test_that_pop_actually_detaches_node() {
        let mut cache = CLruCache::new(5);

        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);
        cache.put("d", 4);
        cache.put("e", 5);

        assert_eq!(cache.pop(&"c"), Some(3));

        cache.put("f", 6);

        let mut iter = cache.iter();
        assert_eq!(iter.next(), Some((&"f", &6)));
        assert_eq!(iter.next(), Some((&"e", &5)));
        assert_eq!(iter.next(), Some((&"d", &4)));
        assert_eq!(iter.next(), Some((&"b", &2)));
        assert_eq!(iter.next(), Some((&"a", &1)));
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_get_with_borrow() {
        let mut cache = CLruCache::new(2);

        let key = String::from("apple");
        cache.put(key, "red");

        assert_eq!(cache.get("apple"), Some(&"red"));
    }

    #[test]
    fn test_get_mut_with_borrow() {
        let mut cache = CLruCache::new(2);

        let key = String::from("apple");
        cache.put(key, "red");

        assert_eq!(cache.get_mut("apple"), Some(&mut "red"));
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_no_memory_leaks() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

        struct DropCounter;

        impl Drop for DropCounter {
            fn drop(&mut self) {
                DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            }
        }

        let n = 100;
        for _ in 0..n {
            let mut cache = CLruCache::new(1);
            for i in 0..n {
                cache.put(i, DropCounter {});
            }
        }
        assert_eq!(DROP_COUNT.load(Ordering::SeqCst), n * n);
    }

    #[test]
    fn test_zero_cap_no_crash() {
        let mut cache = CLruCache::new(0);
        cache.put("key", "value");
    }
}
