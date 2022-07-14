# Snakefile that runs msprime and SLiM simulations
include: "snakefiles/simulations.snake"
include: "snakefiles/slim.snake"
# Snakefile that processes DPGP3 data
include: "snakefiles/dpgp3.snake"

include: "snakefiles/data.snake"
