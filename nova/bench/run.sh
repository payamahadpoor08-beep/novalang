#!/usr/bin/env bash
# Cross-language benchmark. Same algorithm in every language; each must print the
# SAME result (correctness gate) before its time counts. Times are wall-clock ms,
# best of 3 runs. Nova is shown across its tiers: run (interpreter, the oracle),
# vm (bytecode + tiered JIT), aot (native C backend).
#
# Usage: bash run.sh            # all workloads
#        bash run.sh fib        # one workload
set -u
cd "$(dirname "$0")"
NOVA="$(pwd)/../target/release/nova"
WORKLOADS=("${@:-fib sieve mandel}")
[ $# -gt 0 ] && WORKLOADS=("$@") || WORKLOADS=(fib sieve mandel)

have() { command -v "$1" >/dev/null 2>&1; }
# best-of-3 wall time in ms for a command; echoes "ms|firstline_of_output"
timecmd() {
  local best=99999999 out=""
  for _ in 1 2 3; do
    local s e o
    s=$(date +%s%N); o=$("$@" 2>/dev/null); e=$(date +%s%N)
    local ms=$(( (e - s) / 1000000 ))
    [ $ms -lt $best ] && best=$ms
    out="$o"
  done
  echo "${best}|${out%%$'\n'*}"
}

declare -A LANGS
for w in "${WORKLOADS[@]}"; do
  echo "### $w"
  [ -f "$w/main.c" ]    && cc  -O2 -o "$w/c.out"   "$w/main.c"   -lm 2>/dev/null
  [ -f "$w/main.cpp" ]  && c++ -O2 -o "$w/cpp.out" "$w/main.cpp" -lm 2>/dev/null
  [ -f "$w/main.rs" ]   && have rustc && rustc -O -o "$w/rs.out" "$w/main.rs" 2>/dev/null
  [ -f "$w/main.go" ]   && have go && (cd "$w" && go build -o go.out main.go 2>/dev/null)
  [ -f "$w/Main.java" ] && (cd "$w" && javac Main.java 2>/dev/null)

  declare -a rows=()
  expect=""
  add() { # name  ms|out
    local name="$1" r="$2" ms="${2%%|*}" out="${2##*|}"
    if [ -z "$expect" ]; then expect="$out"; fi
    local ok="ok"; [ "$out" = "$expect" ] || ok="MISMATCH($out)"
    rows+=("$name|$ms|$ok")
  }

  have cc        && add "C (gcc -O2)"       "$(timecmd ./$w/c.out)"
  have c++       && add "C++ (g++ -O2)"     "$(timecmd ./$w/cpp.out)"
  [ -x "$w/rs.out" ]     && add "Rust (rustc -O)" "$(timecmd ./$w/rs.out)"
  [ -x "$w/go.out" ]     && add "Go"        "$(timecmd ./$w/go.out)"
  [ -f "$w/Main.class" ] && add "Java"      "$(timecmd java -cp $w Main)"
  # Nova AOT: `nova build` writes ./build/main relative to cwd, so build in-dir
  ( cd "$w" && "$NOVA" build --aot main.nova >/dev/null 2>&1 )
  [ -x "$w/build/main" ] && add "Nova aot (native)" "$(timecmd ./$w/build/main)"
  add "Nova vm (JIT)"      "$(timecmd $NOVA vm "$w/main.nova")"
  have node      && add "JavaScript (node)" "$(timecmd node $w/main.js)"
  have lua5.4    && add "Lua 5.4"           "$(timecmd lua5.4 $w/main.lua)"
  # Node 22 runs TypeScript natively by stripping types
  have node      && add "TypeScript (node --strip-types)" "$(timecmd node --experimental-strip-types $w/main.ts)"
  add "Nova run (interp)"  "$(timecmd $NOVA run "$w/main.nova")"
  have python3   && add "Python 3"          "$(timecmd python3 $w/main.py)"
  have ruby      && add "Ruby"              "$(timecmd ruby $w/main.rb)"

  # sort by ms ascending and print a table
  printf '\n| language | ms | result |\n|---|---:|---|\n'
  for r in "${rows[@]}"; do echo "$r"; done | sort -t'|' -k2 -n | \
    while IFS='|' read -r name ms ok; do printf '| %s | %s | %s |\n' "$name" "$ms" "$ok"; done
  printf '\nexpected result: %s\n\n' "$expect"
done
