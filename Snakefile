# Snakefile that runs msprime and SLiM simulations
# Main simulations
include: "snakefiles/simulations.snake"
include: "snakefiles/slim.snake"
# SI simulations
include: "snakefiles/sequencing_noise.snake"
include: "snakefiles/k_max.snake"
include: "snakefiles/distances.snake"
# DPGP3 data
include: "snakefiles/dpgp3.snake"
include: "snakefiles/data.snake"
