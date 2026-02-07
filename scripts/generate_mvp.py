import argparse
from pathlib import Path

from stm_sim.config import SimulationConfig
from stm_sim.dataset import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Generate STM simulation dataset")
    parser.add_argument("--out", type=str, default="dataset", help="Output directory")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed")
    args = parser.parse_args()

    cfg = SimulationConfig(seed=args.seed)
    generate_dataset(Path(args.out), args.n, cfg, seed=args.seed)


if __name__ == "__main__":
    main()
