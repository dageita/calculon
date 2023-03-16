#!/bin/bash

set -e

export PYTHONPATH=.

# CLI interface infrastructure
echo -e "### Testing top level --help"
./bin/calculon --help > /dev/null
commands=$(./bin/calculon --help | head -n 2 | tail -n 1 | tr '{' ' ' | tr '}' ' ' | tr ',' ' ')
for command in $commands; do
    if [ $command == 'v' ] || [ $command == 'version' ]; then
	echo -e "### Testing \"$command\""
	./bin/calculon $command
    else
	echo -e "### Testing \"$command\" --help"
	./bin/calculon $command --help > /dev/null
    fi
done
echo -e "\n\n"

# Model size calculations
echo -e "### Testing megatron-parameter-calculator"
for model in examples/models/*json; do
    ./bin/calculon megatron-parameter-calculator -a 15 $model
done
echo -e "\n\n"

# Model tests
echo -e "### Testing megatron"
for model in examples/models/*json; do
    echo $model
    ./bin/calculon megatron $model examples/megatron_execution.json examples/a100_80e.json - > /dev/null
    ./bin/calculon megatron $model examples/megatron_execution.json examples/a100_80e.json /tmp/calculon_stats.json -p /tmp/calculon_peers.json
done
echo -e "\n\n"

# Megatron validation
echo -e "### Testing megatron-validation"
./bin/calculon mv -v
echo -e "\n\n"

# Megatron optimal execution
echo -e "### Testing megatron-optimal-execution"
./bin/calculon moe examples/models/turing-530B.json 5128 2520 examples/h100_80g.json -e /tmp/calculon_exe.json -s /tmp/calculon_stats.json -r /tmp/calculon_raw.json
echo -e "\n\n"