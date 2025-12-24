工作将在这台rtx5090主机上进行，推荐使用vscode的ssh进行远程连接。请在ssh config文件中添加如下段落。密码在群里。

```
Host rtx5090
  HostName ssh.cs1.hs1.paratera.com
  User "root@ackcs-00gjfyyh"
  Port 2222
```

之后通过`vscode`的`Remote ssh`功能连接到`rtx5090`主机。

这台主机有1卡32GB显存，120GB内存，Ubuntu24.04系统。当前已经配置好环境`dghnn`，可以通过`conda activate dghnn`命令进入该环境。

所有的工作都将在`root`文件夹中，其中的`DGHNN`文件夹是[这篇文章](https://github.com/skytea/DGHNN)实验的代码，数据集和复现结果。

撰写代码流程：
- fork本仓库
- 在root文件夹中创建自己的文件夹，例如`root/yourname`
- clone fork的仓库到`root/yourname`中
- 在`root/yourname`中进行代码撰写和实验
- push代码到 fork的仓库
- merge到主仓库