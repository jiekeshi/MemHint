# The bugs MemHint found

The 54 memory leaks of Table 1, one row each, with the upstream report they were submitted as.

| Project | Bugs | CodeQL | Infer | Both |
|---|---:|---:|---:|---:|
| Vim | 22 | 18 | 15 | 11 |
| Tmux | 10 | 9 | 6 | 5 |
| OpenSSL | 9 | 4 | 6 | 1 |
| Redis | 3 | 3 | 0 | 0 |
| FreeRDP | 5 | 5 | 2 | 2 |
| curl | 1 | 1 | 1 | 1 |
| FFmpeg | 2 | 2 | 1 | 1 |
| linux/staging | 2 | 2 | 0 | 0 |
| **Total** | **54** | **44** | **31** | **21** |

Status at the time of writing: 54 of 54 confirmed or fixed by maintainers or through submitted patches. A report's GitHub state is recorded but does not decide this: maintainers confirm bugs on threads that stay open, and close threads for reasons other than a fix.

## Vim 9.2.0015 (22)

| Function | Location | Found by | Report | Status |
|---|---|---|---|---|
| `invoke_sync_listeners` | `src/change.c:552` | CodeQL, Infer | [#19516](https://github.com/vim/vim/pull/19516) | fixed |
| `build_drop_cmd` | `src/clientserver.c:703` | CodeQL | [#19517](https://github.com/vim/vim/pull/19517) | fixed |
| `ExpandFromContext` | `src/cmdexpand.c:3457` | CodeQL | [#19505](https://github.com/vim/vim/pull/19505) | fixed |
| `globpath` | `src/cmdexpand.c:4226` | Infer | [#19817](https://github.com/vim/vim/pull/19817) | fixed |
| `eval_dict` | `src/dict.c:960` | CodeQL, Infer | [#19531](https://github.com/vim/vim/pull/19531) | fixed |
| `dict_extend_func` | `src/dict.c:1340` | CodeQL, Infer | [#19518](https://github.com/vim/vim/pull/19518) | fixed |
| `eval8` | `src/eval.c:4787` | Infer | [#19819](https://github.com/vim/vim/pull/19819) | fixed |
| `heredoc_get` | `src/evalvars.c:871` | CodeQL | [#19563](https://github.com/vim/vim/pull/19563) | fixed |
| `ex_substitute` | `src/ex_cmds.c:4489` | CodeQL, Infer | [#19506](https://github.com/vim/vim/pull/19506) | fixed |
| `expand_findfunc` | `src/ex_docmd.c:7076` | CodeQL, Infer | [#19564](https://github.com/vim/vim/pull/19564) | fixed |
| `check_for_cryptkey` | `src/fileio.c:3002` | Infer | [#19821](https://github.com/vim/vim/pull/19821) | fixed |
| `change_indent` | `src/indent.c:1336` | Infer | [#19820](https://github.com/vim/vim/pull/19820) | fixed |
| `list_extend_func` | `src/list.c:2968` | CodeQL, Infer | [#19572](https://github.com/vim/vim/pull/19572) | fixed |
| `add_mark` | `src/mark.c:1475` | CodeQL | [#19827](https://github.com/vim/vim/pull/19827) | fixed |
| `netbeans_file_activated` | `src/netbeans.c:2605` | CodeQL, Infer | [#19626](https://github.com/vim/vim/pull/19626) | fixed |
| `socket_server_send_reply` | `src/os_unix.c:9817` | CodeQL, Infer | [#19627](https://github.com/vim/vim/pull/19627) | fixed |
| `fill_assert_error` | `src/testing.c:155` | CodeQL | [#19507](https://github.com/vim/vim/pull/19507) | fixed |
| `eval_tuple` | `src/tuple.c:512` | CodeQL | [#19628](https://github.com/vim/vim/pull/19628) | fixed |
| `add_interface_from_super_class` | `src/vim9class.c:916` | CodeQL, Infer | [#19629](https://github.com/vim/vim/pull/19629) | fixed |
| `call_oc_method` | `src/vim9class.c:3088` | CodeQL, Infer | [#19828](https://github.com/vim/vim/pull/19828) | fixed |
| `exe_newdict` | `src/vim9execute.c:279` | CodeQL, Infer | [#19508](https://github.com/vim/vim/pull/19508) | fixed |
| `type_name_list_or_dict` | `src/vim9type.c:2554` | CodeQL | [#19513](https://github.com/vim/vim/pull/19513) | fixed |

## Tmux 3.6a (10)

| Function | Location | Found by | Report | Status |
|---|---|---|---|---|
| `cmd_confirm_before_exec` | `cmd-confirm-before.c:95` | Infer | [#4905](https://github.com/tmux/tmux/issues/4905) | fixed |
| `cmd_display_menu_exec` | `cmd-display-menu.c:311` | CodeQL, Infer | [#4875](https://github.com/tmux/tmux/issues/4875) | fixed |
| `cmd_display_popup_exec` | `cmd-display-menu.c:452` | CodeQL | [#4872](https://github.com/tmux/tmux/issues/4872) | fixed |
| `cmd_display_message_exec` | `cmd-display-message.c:80` | CodeQL, Infer | [#4873](https://github.com/tmux/tmux/issues/4873) | fixed |
| `cmd_server_access_exec` | `cmd-server-access.c:88` | CodeQL | [#4876](https://github.com/tmux/tmux/issues/4876) | fixed |
| `cmd_show_prompt_history_exec` | `cmd-show-prompt-history.c:56` | CodeQL | [#4870](https://github.com/tmux/tmux/issues/4870) | fixed |
| `file_get_path` | `file.c:48` | CodeQL, Infer | [#4877](https://github.com/tmux/tmux/issues/4877) | fixed |
| `format_loop_sessions` | `format.c:4430` | CodeQL, Infer | [#4874](https://github.com/tmux/tmux/issues/4874) | fixed |
| `menu_add_item` | `menu.c:84` | CodeQL, Infer | [#4878](https://github.com/tmux/tmux/issues/4878) | fixed |
| `status_prompt_complete_window_menu` | `status.c:1907` | CodeQL | [#4879](https://github.com/tmux/tmux/issues/4879) | fixed |

## OpenSSL 3.6.1 (9)

| Function | Location | Found by | Report | Status |
|---|---|---|---|---|
| `setup_verification_ctx` | `apps/cmp.c:1424` | Infer | [#30377](https://github.com/openssl/openssl/issues/30377) | fixed |
| `app_provider_load` | `apps/lib/app_provider.c:38` | CodeQL, Infer | [#30309](https://github.com/openssl/openssl/issues/30309) | fixed |
| `app_passwd` | `apps/lib/apps.c:236` | CodeQL | [#30308](https://github.com/openssl/openssl/issues/30308) | fixed |
| `load_key_certs_crls` | `apps/lib/apps.c:994` | CodeQL | [#30364](https://github.com/openssl/openssl/issues/30364) | confirmed, thread open |
| `set_up_srp_verifier_file` | `apps/lib/tlssrp_depr.c:192` | CodeQL | [#30362](https://github.com/openssl/openssl/issues/30362) | fixed |
| `EVP_PKEY_derive_SKEY` | `crypto/evp/exchange.c:577` | Infer | [#30378](https://github.com/openssl/openssl/issues/30378) | fixed |
| `X509_policy_check` | `crypto/x509/pcy_tree.c:670` | Infer | [#30435](https://github.com/openssl/openssl/issues/30435) | fixed |
| `port_make_channel` | `ssl/quic/quic_port.c:557` | Infer | [#30440](https://github.com/openssl/openssl/issues/30440) | fixed |
| `dtls1_buffer_message` | `ssl/statem/statem_dtls.c:1175` | Infer | [#30442](https://github.com/openssl/openssl/issues/30442) | fixed |

## Redis 8.6-rc1 (3)

| Function | Location | Found by | Report | Status |
|---|---|---|---|---|
| `lpBatchInsert` | `src/listpack.c:1169` | CodeQL | [#14864](https://github.com/redis/redis/issues/14864) | fixed |
| `fetchClusterConfiguration` | `src/redis-benchmark.c:1234` | CodeQL | [#14861](https://github.com/redis/redis/issues/14861) | fix submitted upstream ([redis#15130](https://github.com/redis/redis/pull/15130), open) |
| `clusterManagerFixSlotsCoverage` | `src/redis-cli.c:6151` | CodeQL | [#14863](https://github.com/redis/redis/pull/14863) | fixed |

## FreeRDP 3.23.0 (5)

| Function | Location | Found by | Report | Status |
|---|---|---|---|---|
| `rdp_client_establish_keys` | `libfreerdp/core/connection.c:760` | CodeQL | [#12487](https://github.com/FreeRDP/FreeRDP/pull/12487) | fixed |
| `freerdp_settings_int_buffer_copy` | `libfreerdp/core/settings.c:1425` | CodeQL, Infer | [#12486](https://github.com/FreeRDP/FreeRDP/pull/12486) | fixed |
| `vgids_read_do_fkt` | `libfreerdp/emu/scard/smartcard_virtual_gids.c:657` | CodeQL, Infer | [#12548](https://github.com/FreeRDP/FreeRDP/pull/12548) | fixed |
| `gdi_create_bitmap` | `libfreerdp/gdi/graphics.c:54` | CodeQL | [#12547](https://github.com/FreeRDP/FreeRDP/pull/12547) | fixed |
| `PCSC_SCardListReadersW` | `winpr/libwinpr/smartcard/smartcard_pcsc.c:1008` | CodeQL | [#12488](https://github.com/FreeRDP/FreeRDP/pull/12488) | fixed |

## curl rc-8\_19\_0 (1)

| Function | Location | Found by | Report | Status |
|---|---|---|---|---|
| `auth_create_digest_http_message` | `lib/vauth/digest.c:728` | CodeQL, Infer | [#20862](https://github.com/curl/curl/pull/20862) | fixed |

## FFmpeg n8.1-dev (2)

| Function | Location | Found by | Report | Status |
|---|---|---|---|---|
| `ffurl_connect` | `libavformat/avio.c:251` | CodeQL | [#22519](https://code.ffmpeg.org/FFmpeg/FFmpeg/pulls/22519) | confirmed, thread open |
| `adapt_colors` | `libswscale/graph.c:516` | CodeQL, Infer | [#22522](https://code.ffmpeg.org/FFmpeg/FFmpeg/pulls/22522) | confirmed, thread open |

## linux/staging v7.0-rc1 (2)

| Function | Location | Found by | Report | Status |
|---|---|---|---|---|
| `atomisp_fixed_pattern_table` | `drivers/staging/media/atomisp/pci/atomisp_cmd.c:3363` | CodeQL | 468454b8 | patch submitted |
| `atomisp_css_allocate_stat_buffers` | `drivers/staging/media/atomisp/pci/atomisp_compat_css20.c:1100` | CodeQL | [12ea5828](https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=12ea58282087b9df896fd0a43a92fa83e56b33c9) | patch merged |

