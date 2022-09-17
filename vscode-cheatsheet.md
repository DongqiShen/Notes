# VSCODE Config

```.vscode```文件夹下包含多个配置文件，包括编译，debug，智能提示等等，都以json格式进行配置。以下分别对每个配置文件的常用设置进行说明。

## task.json
编译配置文件，最外围的```tasks```表示配置列表，其中的每一个元素表示一个配置。
|key|说明|value|
|---|---|---|
|```label```|任务的名称，可以自定义命名|自定义，在```launch.js```中调用这个名字|
|```type```|任务执行的命令类型|```shell```|
|```command```|执行命令的路径|```/usr/bin/clang```|
|```args```|上面命令对应的参数|```"-wall"```,```"-std=c++17"```,```"-o"```,```"${fileDirname}/${fileBasenameNoExtension}"```|
|```detail```|任务的详细描述|自定义，每个任务不一样|

示例如下所示：
 ```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "type": "shell",
            "label": "clang++ build active file",
            "command": "/usr/bin/clang",
            "args": [
                "-std=c++17",
                "-stdlib=libc++",
                "-g",
                "${fileDirname}/*.cpp",
                "-o",
                "${fileDirname}/${fileBasenameNoExtension}",
            ],
            // 以上的部分，就是在shell中执行以下命令，省略了一些参数，最后生成可执行文件filename
            // clang *.cpp -o filename
            "options": {
                "cwd": "${workspaceFolder}"
            },
            "problemMatcher": ["$gcc"], // 使用gcc来捕获错误
            "group": {
                "kind": "build",
                "isDefault": true
            }
            // 任务分组，应为是tasks而不是task，意味着可以执行很多任务
            // 在build组的任务们，可以通过Command Palette(F1)输入run build task来运行
            // 当然，如果任务分组时test，则可以用run test task来执行
        },
    ]
}
 ```

## launch.json
debug配置文件
|key|说明|value|
|---|---|---|
|```name```|任务的名字，按f1可显示|C/C++: clang build and debug active file|
|```preLaunchTask```|在launch之前运行的任务名，一定要和task.json中的任务名一致|clang++ build active file|
|```type```|配置类型|```cppdbg```：表示c++ debug|
|```request```|配置的请求类型|```launch```,```attach```|
|```program```|需要debug的文件名，应和```tasks.json```中保持一致|```${fileDirname}/${fileBasenameNoExtension}```|
|```args```|程序运行时传入的参数列表|[ ]|
|```stopAtEntry```|是否添加断点|默认为```false```，当设置为```true```，程序将在进入```main```函数时打上断点|
|```cmd```|当前工作路径|```${workspaceFolder}```：当前文件所在工作空间|
示例文件如下：
 ```json
 {
    "version": "0.2.0",
    "configurations": [
        {
            "name": "clang++ - Build and debug active file",
            "type": "cppdbg",
            "request": "launch",
            "program": "${fileDirname}/${fileBasenameNoExtension}",
            "args": [],
            "stopAtEntry": true,
            "cwd": "${workspaceFolder}",
            "environment": [],
            "externalConsole": false,
            "MIMode": "lldb",
            "preLaunchTask": "clang++ build active file"
        },
    ]
}
 ```

## 快捷键
常用快捷键设置
|快捷键组合|功能|
|---|---|

