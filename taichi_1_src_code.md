# Taichi code walkthrough

## Kernel compilation
详细分析一下taichik kernel的编译过程，其编译的入口为```Kernel.__call__()```, 下面是编译流程的伪代码：
```python
class Kernel:
    def __call__(self):
        self.ensure_compiled()

    def ensure_compiled(self):
        self.materialize()

    def materialized(self):
        # Field compilation
        self.runtime.materialize()
        # python ast -> fronted ir
        transform_tree(ast_tree, ast_builder)
        # frontend ir -> chi ir -> codegen
        k = prog.create_kernel(ast_builder)
        # prepare runtime executable
        self.get_function_body(k)
```
首先跳过第一步 Field 的编译，它和kernel的编译时两个独立的流程。
### transform_tree(ast_tree, ast_builder)
这一步的目的是将**python ast**转换为**frontend ir**。在```materialized()```里面定义了一个方法```taichi_ast_generator(kernel_cxx)```，而```transform_tree(...)```在里面被调用。在外层调用逻辑上，```taichi_ast_generator(kernel_cxx)```并没有被调用，而是将函数名作为参数放入```prog.create_kernel(...)```中，目前还不太明白这种方法（后续补上），暂且认为```transform_tree(...)```在这里会被调用。这里的调用关系可以是这样子的。```transform_tree(...) -> ASTTransformer()(ctx, tree) -> Builder.__call__(ctx, tree) -> ...```。

    这里有个python的知识，就是当直接调用类名的时候，相当于调用其中的__call__方法，而由于类ASTTransformer继承自Builder，因此相当于直接调用其中的__call__方法。

```Builder.__call__(ctx, node)```会根据参数来决定调用哪个方法。这里有有个python的知识点，就是可以通过构造```string```的方法名从而调用该方法。

    getattr(object, name[, default]) 函数用于返回一个对象属性值，如果定义一个类class A(object): bar = 1, a = A(), 那么getattr(a, 'bar')表示bar的值。在这里则表示的是函数。

这里以调用其中```build_FunctionDef(ctx, node)```为例（```class ASTTransformer```中有很多静态方法，这只是其中的一种）。在该方法内，定义并且调用了```transform_as_kernel()```。这里选择传递pytorch的tensor为例，分析方法调用流程。接着是调用```decl_ndarray_arg()```方法，而在这个方法内，直接调用了c++的代码处理来自pytorch的参数，该方法为```make_external_tensor_expr```。到此为止，python的处理流程告一段落。python会把参数处理的结果封装成一个```class AnyArray```。



## Kernel Execution
taichi的kernel执行的过程主要在```kernel_impl.py```中的```get_function_body(t_kernel)```中实现。简单而言，第一步是对参数的处理，第二部是kernel函数体的编译。taichi中kernel的参数传递的是指针，如果是pytorch的tensor，taichi直接在其原始地址上进行处理，不会进行复制。下面以传递pytorch tensor为例，详细解读整个流程。


```python
def get_function_body(...):
    # 1. Prepare RuntimeContext
    launch_ctx = t_kernel.make_launch_context()
    if ...:
        launch_ctx.set_arg_float(...)
    elif ...:
        launch_ctx.set_arg_external_array_with_shape(...)
    # 2. Execution
        compiled_(launch_ctx);
    # 3. Callback
    if callbacks:
        for c in callbacks:
            c() # for external modules
```
### 1. Prepare RuntimeContext
Todo
#### 2. Execution
整个流程发生在```t_kernel(launch_ctx)```中。这一部分由C++来实现，通过pybind绑定C++函数。值得注意的是，这里有一个python的知识点，就是t_kernel是一个```Class Kernel```的实例，这样的用法实际上是调用了```__call__```函数。因此可以在```export_lang.cpp```中的671行找到这个映射，并且这里调用```kernel->operator()(launch_ctx)```。可以发现```operator()(...)```是```kernel```的一个成员函数。它的实现在```kernel.cpp```的第112行。在这个函数中，可以找到```compiled_(ctx_builder.get_context())```，正好对应于上述代码片段的第二步。