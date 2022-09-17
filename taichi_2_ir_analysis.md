# taichi的 ir分析实战

## python 代码
主要看for循环，省略其他
```python
@ti.kernel
def test_kernel():
    stop = 0
    ti.loop_config(serialize=True)
    for i in range(10):
        if i % 2 == 0:
            continue
        print(i)
        if stop:
            break

test_kernel()
```

## 伪代码
在taichi的的ir实现中，会把for loop转换为while loop。

    i = begin;
    while (1)
    { 
        if (i >= end) {
            break;
        }
        original body;
        i += 1; 
    }

但是这种实现有bug，那就是如果在```original body```中使用了```continue```则会陷入死循环。
所以需要把这种改动一下。

    i = begin - 1;
    while (1) {
        i += 1;
        if (i >= end) {
            break;
        } 
        original body;
    }

## ir代码
部分指令的缩写

    lr -> less than 小于
    ne -> not equal 不等于
    shl -> shirft left 逻辑左移


对应于第一种代码的ir表示如下：

    kernel {
        $0 = offloaded  
        body {
            <i32> $1 = const [0]
            <i32> $2 = const [1]
            <i32> $3 = const [2]
            <i32> $4 = const [10]
            <i32> $5 = alloca
            $6 : while true {
                <i32> $7 = local load [ [$5[0]]]
                <i32> $8 = cmp_lt $7 $4
                $9 : while control nullptr, $8
                <i32> $10 = local load [ [$5[0]]]
                <i32> $11 = div $10 $3
                <i32> $12 = cmp_lt $10 $1
                <i32> $13 = bit_shl $11 $2
                <i32> $14 = cmp_ne $12 $1
                <i32> $15 = cmp_ne $10 $1
                <i32> $16 = cmp_ne $13 $10
                <i32> $17 = bit_and $14 $15
                <i32> $18 = bit_and $17 $16
                <i32> $19 = add $11 $18
                <i32> $20 = bit_shl $19 $2
                <i32> $21 = sub $10 $20
                <i32> $22 = cmp_eq $21 $1
                <i32> $23 = bit_and $22 $2
                $24 : if $23 {
                    $25 continue (scope=$6)
                }
                <i32> $26 = local load [ [$5[0]]]
                print $26, "\n"
                <i32> $28 = add $7 $2
                <i32> $29 : local store [$5 <- $28]
            }
        }
    }

对应于第二种正确的ir表示如下,

    kernel {
        $0 = offloaded  
        body { # 这一部分首先是定义常量
            <i32> $1 = const [0]
            <i32> $2 = const [2]
            <i32> $3 = const [1]
            <i32> $4 = const [-1]
            <i32> $5 = const [10]
            <i32> $6 = alloca # 这里应该loop的变量i
            <i32> $7 : local store [$6 <- $4] # 在进入循环前先-1，并把它放入$6中
            $8 : while true {
                <i32> $9 = local load [ [$6[0]]] # 把变量加载到$9中
                <i32> $10 = add $9 $3 # 变量+1，放入$10
                <i32> $11 = cmp_lt $10 $5 # 变量和循环上限$5比较，是否大于它
                $12 : while control nullptr, $11
                <i32> $13 : local store [$6 <- $10] # 把循环变量再次放入$6中
                <i32> $14 = div $10 $2 # $10中保存的是循环变量的副本，除以2的结果
                <i32> $15 = cmp_lt $10 $1 # $10 和 0 做比较？
                <i32> $16 = bit_shl $14 $3 # 逻辑左移，相当于乘 2
                <i32> $17 = cmp_ne $15 $1 # 循环变量不等于0
                <i32> $18 = cmp_ne $10 $1 # 循环变量不等于0
                <i32> $19 = cmp_ne $16 $10 # # 循环变量不等于0
                <i32> $20 = bit_and $17 $18
                <i32> $21 = bit_and $20 $19
                <i32> $22 = add $14 $21 # 上面的几步很奇怪，先除以2，再乘以2，再判断是不是奇数，如果是，则要把1加上
                <i32> $23 = bit_shl $22 $3 # 乘以2
                <i32> $24 = sub $10 $23 # 循环变量副本要减去一个数
                <i32> $25 = cmp_eq $24 $1 # 上面的结果和0做比较
                <i32> $26 = bit_and $25 $3 # 再和1做比较
                $27 : if $26 { # 如果是满足，则continue，并且跳转到$8
                    $28 continue (scope=$8)
                }
                print $10, "\n"
            }
        }
    }

## 结论
本来想借助这个问题解决一个issuse，目前来对于理解整个流程有一定的帮助，但是对于解决这个问题，还是有点距离。low_ast.cpp主要是把taichi ir转换为chi ir，但这都是处于编译阶段。在源码中，根据for loop的实现，它是写死的指令，默认从begin（小数）开始，每次循环+1，直到end。然而，range(n, m)中这两个是变量，而不是常量，它们是在运行时决定的，因此，目前我的理解是，如果需要支持的话，在转换为chi ir的阶段，应该要写很多代码，我还需要先去了解一下在clang前端中，是如何实现for循环的。