#!/usr/bin/env bash
# Check out the eight subject projects at the exact commits the paper analysed.
#
#     ./experiment/fetch_subjects.sh                 all eight, into experiment/subjects/
#     ./experiment/fetch_subjects.sh tmux_3_6_a      one of them
#     SUBJECTS=/data/subjects ./experiment/fetch_subjects.sh
#
# Each project is fetched at depth 1 at a pinned commit, so the checkout is the
# same tree the paper's measurements were taken on regardless of what upstream
# has tagged since. The tag column of experiment/projects.md is the human name
# of the same commit. The Linux kernel is the largest download (about 300 MB
# compressed at depth 1); only drivers/staging is analysed.
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBJECTS="${SUBJECTS:-$HERE/subjects}"

# directory  repository  commit  (tag, for the reader)
read -r -d '' TABLE <<'EOF'
vim_9_2_0015      https://github.com/vim/vim.git           b8b7df29c418b93fa3d8607ae22f10701fa168d6  v9.2.0015
tmux_3_6_a        https://github.com/tmux/tmux.git         cc117b5048f77a4842820f8ebbe3a86e5c077224  3.6a
openssl_3_6_1     https://github.com/openssl/openssl.git   c9a9e5b10105ad850b6e4d1122c645c67767c341  openssl-3.6.1
redis-8_6-rc1     https://github.com/redis/redis.git       56020bef968b33a15437649dccbee3dfae28901a  8.6-rc1
FreeRDP_3_23_0    https://github.com/FreeRDP/FreeRDP.git   b933ae18d9ad2a1d73c610868fcc30eb61654070  3.23.0
curl_rc-8_19_0-3  https://github.com/curl/curl.git         a67ee591e05c025641ca547e7398b2e0ebcbba6a  8.19.0-rc3
FFmpeg_n_8_1_dev  https://github.com/FFmpeg/FFmpeg.git     638b521c7b09e00514aa13ade43c389b4b40ddfd  n8.1-dev
linux_v7_0-rc1    https://github.com/torvalds/linux.git    6de23f81a5e08be8fbf5e8d7e9febc72a5b5f27f  v7.0-rc1
EOF

command -v git >/dev/null || { echo "!! git is not on PATH"; exit 1; }
mkdir -p "$SUBJECTS"

wanted=("$@")
status=0
while read -r dir url commit tag; do
    [ -z "$dir" ] && continue
    if [ ${#wanted[@]} -gt 0 ]; then
        keep=0
        for w in "${wanted[@]}"; do [ "$w" = "$dir" ] && keep=1; done
        [ "$keep" = 1 ] || continue
    fi
    dest="$SUBJECTS/$dir"
    if [ -d "$dest/.git" ]; then
        have="$(git -C "$dest" rev-parse HEAD 2>/dev/null)"
        if [ "$have" = "$commit" ]; then
            echo "== $dir: already at $tag ($commit)"
            continue
        fi
        echo "!! $dir exists at $have, not $commit; remove it to refetch"
        status=1
        continue
    fi
    echo "== $dir: fetching $tag ($commit)"
    mkdir -p "$dest"
    if git -C "$dest" init -q \
        && git -C "$dest" remote add origin "$url" \
        && git -C "$dest" fetch -q --depth 1 origin "$commit" \
        && git -C "$dest" checkout -q FETCH_HEAD; then
        echo "   ok"
    else
        echo "!! $dir: fetch failed"
        rm -rf "$dest"
        status=1
    fi
done <<< "$TABLE"

if [ $status -eq 0 ]; then
    echo
    echo "Subjects are under $SUBJECTS. Build dependencies are listed in README.md (Installation)."
fi
exit $status
