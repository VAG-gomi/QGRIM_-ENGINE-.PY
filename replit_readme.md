# Workspace

## Overview

pnpm workspace monorepo using TypeScript. Each package manages its own dependencies.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally

See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details.

## QGRIM Blueprint (`qgrim/`)

A standalone hardware blueprint package for the QGRIM v2.1 FPGA quantum
circuit simulator (4-qubit state-vector simulator on iCE40HX1K). Contains:

- `qgrim/rtl/` — synthesizable Verilog (12 modules)
- `qgrim/host/` — Python assembler, SPI driver, software model
- `qgrim/sim/` — Cocotb testbenches (Bell state, Born-rule measurement)
- `qgrim/docs/` — register map, ISA, architecture
- `qgrim/Makefile` — yosys + nextpnr-ice40 + icepack flow
- `qgrim/qgrim.pcf` — iCEstick pin constraints

Three structural fixes vs the v2.0 audit are implemented in
`spi_bram_bridge.v`, `measurement_unit.v`, and `pair_addr_gen.v`.

Verify the software model: `cd qgrim/host && python3 qgrim_sim.py examples/bell.qasm`
