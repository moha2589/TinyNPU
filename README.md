# TinyNPU

A tiny neural network accelerator built with [Stone Arch Silicon](https://stone-arch-silicon.github.io/stone-arch-silicon/), the ASIC and chip design club at the University of Minnesota.

TinyNPU is a small chip that speeds up the math behind neural networks. We’re designing it from scratch and having it made through [Tiny Tapeout](https://tinytapeout.com). The heart of the project is the systolic array, a small square grid of processing elements. Each processing element recieves an two 8-bit numbers, multiplies them against its own weight, then adds that to a running total. To learn more about the project, see #Architecture

## Status

| Part | Owner | Status |
|---|---|---|
| Quantized MNIST model | ML |
| Golden model | Verification |
| PE + systolic array RTL | RTL |
| SPI + control FSM | RTL |
| cocotb testbench | Verification |
| Hardening + timing | Physical design |

## Architecture



## Repo layout

| Folder | What's in it |
|---|---|
| `src/` | Verilog RTL. |
| `test/` | cocotb testbenches. |
| `golden/` | Bit-exact Python model of the chip. All tests check against it. |
| `model/` | PyTorch training for weights |
| `firmware/` | RP2040 firmware. |
| `docs/` | Specs, pinout and meeting notes.  |


See [`docs/info.md`](docs/info.md) and `info.yaml`.

## Team

RTL: _name_ · Verification: _name_ · Physical design: _name_ · Analog: _name_ · ML: _name_

## License

Apache 2.0 (from the Tiny Tapeout template).