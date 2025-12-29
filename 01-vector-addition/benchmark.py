import torch

from Triton.native import solve as triton_add
from Tilelang.native import solve as tile_add
from Tilelang.cutedsl import solve as tile_cutedsl_add

from importlib.metadata import version
from importlib.util import find_spec


def benchmark(f, n=None, nvtx=None, *args, **kwargs):
    # trigger jit
    f(*args, **kwargs)

    if n is None:
        assert False, "TODO: estimate time"

    if nvtx:
        torch.cuda.nvtx.range_push(nvtx)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for _ in range(n):
        f(*args, **kwargs)

    end.record()
    end.synchronize()

    if nvtx:
        torch.cuda.nvtx.range_pop()

    return start.elapsed_time(end)


sizes = [128 * 2**p for p in range(0, 10)]
sizes = [127, 129] + sizes

tests = {
    "Triton": triton_add,
    "Tilelang": tile_add,
    "Torch": lambda a, b, c, size: torch.add(a, b, out=c),
}


if find_spec("cutlass") is None:
    # CuTeDSL not installed
    pass
elif version("nvidia-cutlass-dsl") >= "4.3.4":
    import cutlass.cute as cute
    from CuTeDSL.native import solve as cute_kernel

    def cutedsl_add(
        a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, n: int, cache={}
    ):
        if n not in cache:
            a_fake = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
            b_fake = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))
            c_fake = cute.runtime.make_fake_compact_tensor(cute.Float32, (n,))

            cache[n] = cute.compile(
                cute_kernel, a_fake, b_fake, c_fake, n, options="--enable-tvm-ffi"
            )

        cache[n](a, b, c, n)

    tests["CuTeDSL"] = cutedsl_add
else:
    tests["Tilelang (CuTeDSL)"] = lambda a, b, c, size: tile_cutedsl_add(a, b, c, size)


results = {}


for i, size in enumerate(sizes):
    a = torch.randn(size, device="cuda")
    b = torch.randn(size, device="cuda")
    c = torch.zeros(size, device="cuda")

    triton_add(a, b, c, size)

    assert torch.allclose(c, a + b)

    for title, test in tests.items():
        results.setdefault(title, []).append(
            benchmark(lambda: test(a, b, c, size), 100)
        )


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
ax = plt.gca()

for title, result in results.items():
    sns.regplot(
        x="x",
        y="y",
        data=pd.DataFrame({"x": sizes, "y": result}),
        ax=ax,
        label=title,
        scatter_kws={"s": 50},
    )

plt.title("Vectorized add")
plt.xlabel("Size")
plt.ylabel("Time (ms)")

plt.legend()

plt.savefig("01-result.png")
