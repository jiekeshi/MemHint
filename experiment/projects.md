# Subject projects

The eight projects of Table 1, at the commits the paper analysed. `fetch_subjects.sh` checks them out into `experiment/subjects/` (or `$SUBJECTS`) under these directory names, which the build configurations in `../proj_build_command_codeql.json` and `../proj_build_command_infer.json` are keyed by.

| Directory | Project | Version | Commit |
|---|---|---|---|
| `vim_9_2_0015` | Vim | 9.2.0015 | `b8b7df29c418b93fa3d8607ae22f10701fa168d6` |
| `tmux_3_6_a` | tmux | 3.6a | `cc117b5048f77a4842820f8ebbe3a86e5c077224` |
| `openssl_3_6_1` | OpenSSL | 3.6.1 | `c9a9e5b10105ad850b6e4d1122c645c67767c341` |
| `redis-8_6-rc1` | Redis | 8.6-rc1 | `56020bef968b33a15437649dccbee3dfae28901a` |
| `FreeRDP_3_23_0` | FreeRDP | 3.23.0 | `b933ae18d9ad2a1d73c610868fcc30eb61654070` |
| `curl_rc-8_19_0-3` | curl | 8.19.0-rc3 | `a67ee591e05c025641ca547e7398b2e0ebcbba6a` |
| `FFmpeg_n_8_1_dev` | FFmpeg | n8.1-dev | `638b521c7b09e00514aa13ade43c389b4b40ddfd` |
| `linux_v7_0-rc1` | Linux `drivers/staging` | v7.0-rc1 | `6de23f81a5e08be8fbf5e8d7e9febc72a5b5f27f` |

Only the `drivers/staging` subsystem of the kernel is parsed and analysed (`run.sh` passes `--source-root drivers/staging`); the kernel configuration steps in the build file exist so that the subsystem compiles.

## What gets compiled

Stage 1 parses every C/C++ file under the project (or the source root). The analyzers of Stage 2 see what the build compiles. Each project is built with its default configuration, adding only what the build needs to run: curl's configure requires a TLS backend to be chosen (`--with-openssl`); Redis is built for Infer with `OPT=-fno-lto`, since Infer's clang cannot link the LTO objects of the default build; OpenSSL is configured with `no-tests` for CodeQL, so that the database holds the library and apps rather than the test programs; and the kernel needs `allmodconfig` with the signing-key options cleared and `clang-17`. The apt command in `README.md` installs the libraries these default builds detect. The SLOC and function counts of Table 1 describe the source trees Stage 1 parsed.
