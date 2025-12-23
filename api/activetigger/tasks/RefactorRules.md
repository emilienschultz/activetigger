# Rules for refactoring

1. Names of the functions for refactoring

- `__init_paths`: First function to check the existance of elements, remove folders if necessary and so on
- `__load_*`: given a path or parameters load
- `__check_*`: check the right format of functions
- `__compute_*`: For heavy tasks (like embedding generation)
- `__create_*`: when writing files like `__create_report`

2. Every object attributes should be instantiated and updated in the `__call__` function.
3. The main narative should appear in the `__call__` function. <br/>
   _Example: to load or compute embeddings, have 2 steps:_

```python
compute_embeddings: bool = self.__check_embeddings(paths...)
if compute_embeddings:
    self.__compute_emebddings(...)
self.__load_embeddings(....)
```

4. Write docstrings
