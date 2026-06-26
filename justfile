set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command"]

python := if os() == "windows" { "./.venv/Scripts/python.exe" } else { "./.venv/bin/python" }

default:
    just --list

convert *args:
    {{python}} src/cli.py convert {{args}}

tag *args:
    {{python}} src/cli.py tag {{args}}

rename *args:
    {{python}} src/cli.py rename {{args}}

shelve *args:
    {{python}} src/cli.py shelve {{args}}

gallery *args:
    {{python}} src/cli.py gallery {{args}}

wall *args:
    {{python}} src/cli.py wall {{args}}

run: convert tag rename gallery

# test tasks

scramble *args:
    {{python}} src/cli.py scramble {{args}}

clean *args:
    {{python}} src/cli.py clean {{args}}

test: clean scramble run
