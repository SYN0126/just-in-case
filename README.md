# CUDA Operators

CUDA 算子与 GEMM 实验代码。

## 目录

- `GEMM/`：矩阵乘法相关的 CUDA 源代码。
- `上传 GitHub.command`：macOS 双击上传脚本。

## 上传到 GitHub

1. 将算子代码放进本目录，可以按算子新建子文件夹。
2. 开启代理软件及其系统代理；可以保持规则 / 条件模式。
3. 双击根目录的 **上传 GitHub.command**。
4. 首次运行填写公开仓库名，默认 `just-in-case`；之后填写修改说明并上传。说明留空会自动使用时间。

脚本使用当前 `gh` 登录账号，首次创建 **公开仓库**，之后同步到已关联的仓库。上传包含本目录及子目录中的新增、修改和删除，`.gitignore` 中列出的内容除外。

网络配置优先使用终端代理环境变量，否则读取 macOS 系统的 HTTPS、HTTP 或 SOCKS 代理。设置只在脚本运行期间生效。规则模式仍需要代理软件允许访问 GitHub；仅开启 PAC 而没有手动系统代理时，可在终端设置 `HTTPS_PROXY` 后运行脚本。

已安装 Git 与 GitHub CLI 后，首次登录可在终端运行：

```bash
gh auth login --hostname github.com --git-protocol https --web
```

检查 API 和 Git 连接（只读，不上传）：

```bash
./"上传 GitHub.command" --check
```

上传失败时，本地提交会保留，再次双击可重试。若远端出现其他提交，脚本会提示先合并；它不会强制覆盖远端历史。

参考：[GitHub CLI 创建仓库](https://cli.github.com/manual/gh_repo_create)、[Git 的代理配置](https://git-scm.com/docs/git-config#Documentation/git-config.txt-httpproxy)。
