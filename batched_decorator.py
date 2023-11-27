from common_imports import *

class batched:
    """
    A decorator to run a function in batches over given arguments. The results
    from each batch are aggregated using a reducer function, e.g. sum, mean, or
    concatenation.
    
    Things that came up during use:
    - sometimes, you return a list of things, and you want to concatenate across
    respective elements of the list, instead of concatenating all the lists into
    one big list.
    - sometimes you return a variable number of outputs
    - sometimes it is more natural to concatenate over a dimension different 
    from the first one.
    - sometimes you want to concatenate dataframes instead of tensors.
    
    """

    def __init__(
        self,
        args: List[str],
        n_outputs: Union[int, Literal['var']],
        reducer: Union[Callable, str] = "cat",
        shuffle: bool = False,
        verbose: bool = True,
    ):
        self.args = args
        self.n_outputs = n_outputs
        self.reducer = reducer
        self.shuffle = shuffle
        self.verbose = verbose
        if self.shuffle:
            raise NotImplementedError
    
    T = typing.TypeVar("T", Tensor, np.ndarray, Sequence)
    @staticmethod
    def get_slice(x: T, idx: np.ndarray) -> T:
        if isinstance(x, (Tensor, np.ndarray)):
            return x[idx]
        elif isinstance(x, (list, tuple)):
            return type(x)([x[i] for i in idx])
        else:
            try:
                return x[idx]
            except:
                raise NotImplementedError(f"Cannot slice {type(x)}")
    
    @staticmethod
    def average_objs(xs: List[T], dim: int = 0) -> Union[T, Dict[Any, T], List[T]]:
        assert len({type(x) for x in xs}) == 1
        if isinstance(xs[0], (Tensor, np.ndarray)):
            return sum(xs) / len(xs)
        elif isinstance(xs[0], pd.DataFrame):
            return sum(xs) / len(xs)
        elif isinstance(xs[0], list):
            assert len({len(x) for x in xs}) == 1
            return [batched.average_objs([x[i] for x in xs], dim=dim) for i in range(len(xs[0]))]
        elif isinstance(xs[0], dict):
            # check all dicts have the same set of keys
            assert all(set(x.keys()) == set(xs[0].keys()) for x in xs)
            return {k: batched.average_objs([x[k] for x in xs], dim=dim) for k in xs[0].keys()}
        elif xs[0] is None:
            return None
        else:
            raise NotImplementedError
        
    @staticmethod
    def concatenate_objs(xs: Any, dim: int = 0) -> Any:
        assert len({type(x) for x in xs}) == 1
        # if isinstance(xs[0], TransientObj):
        #     return Transient(batched.concatenate_objs([x.obj for x in xs], dim=dim))
        if isinstance(xs[0], Tensor):
            return torch.cat(xs, dim=dim)
        elif isinstance(xs[0], np.ndarray):
            return np.concatenate(xs, axis=dim)
        elif isinstance(xs[0], pd.DataFrame):
            return pd.concat(xs, ignore_index=True)
        elif isinstance(xs[0], dict):
            # check all dicts have the same set of keys
            assert all(set(x.keys()) == set(xs[0].keys()) for x in xs)
            return {k: batched.concatenate_objs([x[k] for x in xs], dim=dim) for k in xs[0].keys()}
        elif isinstance(xs[0], list):
            assert len({len(x) for x in xs}) == 1
            return [batched.concatenate_objs([x[i] for x in xs], dim=dim) for i in range(len(xs[0]))]
        elif xs[0] is None:
            return None
        else:
            raise NotImplementedError

    def __call__(self, func: Callable) -> "func":
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            batch_size = kwargs.get("batch_size", None)
            if batch_size is None:
                return func(*args, **kwargs)
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()
            named_args = dict(bound_args.arguments)
            batching_args = {k: named_args[k] for k in self.args}
            # check all the lengths are the same
            lengths = [len(v) for v in batching_args.values()]
            assert (
                len(set(lengths)) == 1
            ), "All batched arguments must have the same length."
            length = lengths[0]
            assert length > 0
            num_batches = math.ceil(length / batch_size)
            results = []
            pbar = tqdm if self.verbose else lambda x: x
            for i in pbar(range(num_batches)):
                batch_idx = np.arange(
                    i * batch_size, min(lengths[0], (i + 1) * batch_size)
                )
                batched_args = {k: batched.get_slice(v, batch_idx) for k, v in batching_args.items()}
                named_args.update(batched_args)
                results.append(func(**named_args))
            # todo: refactor this logit to be uniform across reducers
            if self.reducer.startswith('cat'):
                if self.reducer == 'cat':
                    dim = 0
                else:
                    _, dim = self.reducer.split('_')
                    dim = int(dim)
                # concatenate the results per output
                if self.n_outputs == 1:
                    return batched.concatenate_objs(results, dim=dim)
                else:
                    assert len({len(r) for r in results}) == 1
                    return tuple([
                        batched.concatenate_objs([r[i] for r in results], dim=dim)
                        for i in range(len(results[0]))
                    ])
            elif self.reducer == "mean":
                if self.n_outputs == 1:
                    return batched.average_objs(results)
                else:
                    assert len({len(r) for r in results}) == 1
                    return tuple([
                        sum([r[i] for r in results]) / len(results)
                        for i in range(len(results[0]))
                    ])
            else:
                raise NotImplementedError

        return wrapper