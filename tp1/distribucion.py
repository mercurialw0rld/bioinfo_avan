"""Comparación de proteínas humanas reales y secuencias proteicas al azar.

Dependencias: biopython, matplotlib y numpy.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_SEED = 42


def downloadHumanProteins(
    email: str,
    numberOfProteins: int = 1000,
    outputPath: Path = Path(__file__).with_name("proteinas_humanas.fasta"),
    seed: int = DEFAULT_SEED,
    searchPoolSize: int = 10_000,
) -> list[SeqRecord]:
    """Descarga una muestra aleatoria de proteínas humanas de NCBI Protein.

    La búsqueda excluye secuencias parciales y selecciona IDs al azar desde un
    conjunto de hasta ``searchPoolSize`` resultados. El FASTA se usa como caché.
    """
    if numberOfProteins <= 0:
        raise ValueError("La cantidad de proteínas debe ser mayor que cero.")
    if searchPoolSize < numberOfProteins:
        raise ValueError("El conjunto de búsqueda debe ser mayor que la muestra.")
    if outputPath.exists():
        cachedRecords = list(SeqIO.parse(outputPath, "fasta"))
        if len(cachedRecords) >= numberOfProteins:
            return cachedRecords[:numberOfProteins]

    Entrez.email = email
    searchTerm = 'Homo sapiens[Organism] AND srcdb_refseq[PROP] NOT partial[Title]'
    with Entrez.esearch(
        db="protein", term=searchTerm, retmax=searchPoolSize, usehistory="n"
    ) as searchHandle:
        proteinIds = Entrez.read(searchHandle)["IdList"]

    if len(proteinIds) < numberOfProteins:
        raise ValueError(
            f"NCBI devolvió {len(proteinIds)} proteínas; se solicitaron "
            f"{numberOfProteins}."
        )

    randomGenerator = random.Random(seed)
    selectedIds = randomGenerator.sample(proteinIds, numberOfProteins)
    records: list[SeqRecord] = []

    # NCBI recomienda dividir las descargas grandes en lotes.
    for batchStart in range(0, len(selectedIds), 200):
        batchIds = selectedIds[batchStart : batchStart + 200]
        with Entrez.efetch(
            db="protein", id=",".join(batchIds), rettype="fasta", retmode="text"
        ) as fastaHandle:
            records.extend(SeqIO.parse(fastaHandle, "fasta"))
        sleep(0.34)

    outputPath.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, outputPath, "fasta")
    return records


def generateRandomProteins(
    realProteins: list[SeqRecord], seed: int = DEFAULT_SEED
) -> list[SeqRecord]:
    """Genera proteínas equiprobables con las mismas longitudes que las reales."""
    randomGenerator = random.Random(seed)
    randomProteins = []
    for index, realProtein in enumerate(realProteins, start=1):
        sequence = "".join(
            randomGenerator.choices(AMINO_ACIDS, k=len(realProtein.seq))
        )
        randomProteins.append(
            SeqRecord(Seq(sequence), id=f"random_{index}", description="")
        )
    return randomProteins


def aminoAcidDistribution(proteins: list[SeqRecord]) -> np.ndarray:
    """Devuelve las frecuencias relativas de los 20 aminoácidos estándar."""
    counts = Counter(
        aminoAcid
        for protein in proteins
        for aminoAcid in str(protein.seq).upper()
        if aminoAcid in AMINO_ACIDS
    )
    total = sum(counts.values())
    if total == 0:
        return np.zeros(len(AMINO_ACIDS))
    return np.array([counts[aminoAcid] / total for aminoAcid in AMINO_ACIDS])


def cumulativeDistributionRmse(proteins: list[SeqRecord]) -> np.ndarray:
    """Calcula el RMSE entre cada distribución acumulada y la distribución final."""
    if not proteins:
        return np.array([])

    finalDistribution = aminoAcidDistribution(proteins)
    cumulativeCounts = Counter()
    total = 0
    rmseValues = []

    for protein in proteins:
        validAminoAcids = [
            aminoAcid
            for aminoAcid in str(protein.seq).upper()
            if aminoAcid in AMINO_ACIDS
        ]
        cumulativeCounts.update(validAminoAcids)
        total += len(validAminoAcids)
        if total == 0:
            currentDistribution = np.zeros(len(AMINO_ACIDS))
        else:
            currentDistribution = np.array(
                [cumulativeCounts[aminoAcid] / total for aminoAcid in AMINO_ACIDS]
            )
        rmseValues.append(np.sqrt(np.mean((currentDistribution - finalDistribution) ** 2)))

    return np.array(rmseValues)


def createPlots(
    realProteins: list[SeqRecord],
    randomProteins: list[SeqRecord],
    outputPath: Path = Path("comparacion_distribuciones.png"),
) -> None:
    """Crea y guarda los tres gráficos solicitados en una sola figura."""
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    figure.patch.set_facecolor("#f7f7f5")
    realColor = "#2a6f97"
    randomColor = "#e07a5f"

    positions = np.arange(len(AMINO_ACIDS))
    barWidth = 0.42
    realDistribution = aminoAcidDistribution(realProteins)
    randomDistribution = aminoAcidDistribution(randomProteins)
    axes[0].bar(positions - barWidth / 2, realDistribution, barWidth,
                label="Reales", color=realColor, alpha=0.9)
    axes[0].bar(positions + barWidth / 2, randomDistribution, barWidth,
                label="Aleatorias", color=randomColor, alpha=0.85)
    axes[0].set_xticks(positions, AMINO_ACIDS)
    axes[0].set_ylabel("Frecuencia relativa")
    axes[0].set_title("Composición de aminoácidos", fontweight="bold")
    axes[0].legend(frameon=False)

    realLengths = [len(protein.seq) for protein in realProteins]
    axes[1].hist(realLengths, bins="fd", color=realColor, alpha=0.85,
                 edgecolor="white")
    axes[1].axvline(np.median(realLengths), color="#243447", linestyle="--",
                    label=f"Mediana: {np.median(realLengths):.0f} aa")
    axes[1].set_xlabel("Longitud (aminoácidos)")
    axes[1].set_ylabel("Cantidad de proteínas")
    axes[1].set_title("Longitudes de proteínas reales", fontweight="bold")
    axes[1].legend(frameon=False)

    realRmse = cumulativeDistributionRmse(realProteins)
    randomRmse = cumulativeDistributionRmse(randomProteins)
    proteinNumbers = np.arange(1, len(realProteins) + 1)
    axes[2].plot(proteinNumbers, realRmse, color=realColor, label="Reales", lw=2)
    axes[2].plot(proteinNumbers, randomRmse, color=randomColor,
                 label="Aleatorias", lw=2)
    axes[2].set_xlabel("Número de proteínas acumuladas")
    axes[2].set_ylabel("RMSE respecto de la distribución final")
    axes[2].set_title("Estabilización de la distribución", fontweight="bold")
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)

    figure.suptitle("Proteínas humanas vs. secuencias al azar", fontsize=17,
                    fontweight="bold", color="#243447")
    figure.tight_layout()
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(outputPath, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True, help="Email requerido por NCBI Entrez")
    parser.add_argument("--cantidad", type=int, default=1000)
    parser.add_argument("--semilla", type=int, default=DEFAULT_SEED)
    parser.add_argument("--salida", type=Path, default=Path("comparacion_distribuciones.png"))
    arguments = parser.parse_args()

    realProteins = downloadHumanProteins(
        arguments.email, arguments.cantidad, seed=arguments.semilla
    )
    randomProteins = generateRandomProteins(realProteins, arguments.semilla)
    createPlots(realProteins, randomProteins, arguments.salida)
    print(f"Gráfico guardado en: {arguments.salida.resolve()}")


if __name__ == "__main__":
    main()
