#!/usr/bin/env bash
# Fetch the public external benchmarks used by E1/E2 into this directory.
# We do NOT redistribute the third-party data in the repo; this script pulls it
# from the original sources on demand. Cite the original papers, not this repo.
#
#   HHCRSP (E1): Mankowska, Meisel & Bierwirth (2014), Health Care Management
#     Science 17(1):15-30, doi:10.1007/s10729-013-9243-1.
#     Source page: https://prodlog.wiwi.uni-halle.de/forschung/research_data/hhcrsp/
#   SMPTSP (E2, future work): Krishnamoorthy/Ernst/Baatar; Smet et al.
#     Sources: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/ptaskinfo.html
#              https://people.cs.kuleuven.be/~pieter.smet/smptsp.html
#
#   usage:  cd Code/journal_p0/external/data && bash download.sh
set -euo pipefail
cd "$(dirname "$0")"
UA="research/1.0 (benchmark fetch for reproducibility)"

echo "[1/2] HHCRSP (Mankowska et al. 2014) -> hhcrsp/"
mkdir -p hhcrsp
# Halle download endpoints (as of 2026-07). If these break, download the instances
# zip + 'Description of file format.pdf' manually from the source page above.
curl -fsSL -A "$UA" -o hhcrsp/_instances.zip \
  "https://wcms.itz.uni-halle.de/download.php?down=38055&elem=2882666"
curl -fsSL -A "$UA" -o hhcrsp/HHCRSP_file_format.pdf \
  "https://wcms.itz.uni-halle.de/download.php?down=38056&elem=2882667" || true
( cd hhcrsp && unzip -o _instances.zip >/dev/null && rm -f _instances.zip )
echo "  got $(ls hhcrsp/Instanz*.txt 2>/dev/null | wc -l | tr -d ' ') HHCRSP instance files"

echo "[2/2] SMPTSP / PTRP (Smet et al.) -> smptsp/"
mkdir -p smptsp
curl -fsSL -A "$UA" -o smptsp/_smptsp.zip \
  "https://people.cs.kuleuven.be/~pieter.smet/smptsp.zip"
( cd smptsp && unzip -o _smptsp.zip >/dev/null && rm -f _smptsp.zip )
echo "  got $(ls smptsp/*.dat 2>/dev/null | wc -l | tr -d ' ') SMPTSP instance files"

echo "done. Reproduce Table tab:external with:"
echo "  cd ../../..  &&  python3 journal_p0/external/run_family.py --benchmark hhcrsp \\"
echo "    --glob 'journal_p0/external/data/hhcrsp/InstanzCPLEX_HCSRP_75_*.txt' --rho-sweep 0.55,0.65,0.75,0.85 --slots 3"
