"""Detector de ORFs y clasificador bayesiano de regiones codificantes.

El conjunto codificante C se construye con proteinas humanas reales de
``distribucion.py``. El conjunto aleatorio R se obtiene al generar ADN uniforme,
buscar ORFs con el mismo detector y traducirlos.

Dependencias: biopython, matplotlib y numpy.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from matplotlib.ticker import PercentFormatter

from distribucion import AMINO_ACIDS, downloadHumanProteins


DNA_BASES = "ACGT"
START_CODON = "ATG"
STOP_CODONS = {"TAA", "TAG", "TGA"}
DEFAULT_SEED = 42
SMOOTHING = 1.0


@dataclass
class Orf:
    """ORF detectado en una hebra orientada 5'->3'."""

    strand: str
    frame: int
    start: int
    end: int
    dnaSequence: Seq
    proteinSequence: Seq

    @property
    def aminoAcidLength(self) -> int:
        """Longitud proteica, sin contar el codon de stop."""
        return len(self.proteinSequence)


@dataclass
class LengthModel:
    """Probabilidades suavizadas de los bins de longitud para C y R."""

    binEdges: np.ndarray
    codingProbabilities: np.ndarray
    randomProbabilities: np.ndarray

    def getBinIndex(self, length: int) -> int:
        """Asigna longitudes fuera del rango al bin extremo correspondiente."""
        index = np.searchsorted(self.binEdges, length, side="right") - 1
        return int(np.clip(index, 0, len(self.codingProbabilities) - 1))


@dataclass
class OrfClassifier:
    """Modelo generativo P(L, composicion | C/R) con priors iguales por defecto."""

    lengthModel: LengthModel
    codingAminoAcidProbabilities: np.ndarray
    randomAminoAcidProbabilities: np.ndarray
    codingPrior: float = 0.5
    randomPrior: float = 0.5

    def logLikelihoodRatio(self, proteinSequence: Seq) -> float:
        """Calcula S = log prior odds + log P(L|C)/P(L|R) + termino multinomial."""
        counts = getAminoAcidCounts(proteinSequence)
        length = int(counts.sum())
        lengthBin = self.lengthModel.getBinIndex(length)

        priorOdds = math.log(self.codingPrior / self.randomPrior)
        lengthOdds = math.log(
            self.lengthModel.codingProbabilities[lengthBin]
            / self.lengthModel.randomProbabilities[lengthBin]
        )
        compositionOdds = float(
            np.sum(
                counts * np.log(
                    self.codingAminoAcidProbabilities
                    / self.randomAminoAcidProbabilities
                )
            )
        )
        return priorOdds + lengthOdds + compositionOdds

    def codingProbability(self, proteinSequence: Seq) -> float:
        """Convierte S a P(C|L, n) de forma numericamente estable."""
        score = self.logLikelihoodRatio(proteinSequence)
        if score >= 0:
            return 1 / (1 + math.exp(-score))
        exponentialScore = math.exp(score)
        return exponentialScore / (1 + exponentialScore)


def downloadHumanGene(
    email: str, geneName: str | None = None, seed: int = DEFAULT_SEED
) -> SeqRecord:
    """Descarga desde RefSeqGene un gen humano indicado o uno aleatorio."""
    Entrez.email = email
    baseQuery = 'Homo sapiens[Organism] AND refseqgene[filter]'
    searchQuery = f'{baseQuery} AND "{geneName}"[Gene Name]' if geneName else baseQuery

    with Entrez.esearch(db="nuccore", term=searchQuery, retmax=500) as searchHandle:
        nucleotideIds = Entrez.read(searchHandle)["IdList"]
    if not nucleotideIds:
        requestedName = f" para el gen {geneName!r}" if geneName else ""
        raise ValueError(f"No se encontro una entrada RefSeqGene humana{requestedName}.")

    selectedId = nucleotideIds[0] if geneName else random.Random(seed).choice(nucleotideIds)
    with Entrez.efetch(
        db="nuccore", id=selectedId, rettype="fasta", retmode="text"
    ) as fastaHandle:
        return SeqIO.read(fastaHandle, "fasta")


def getReadingFrames(dnaSequence: Seq) -> list[tuple[str, int, Seq]]:
    """Devuelve los tres marcos de las dos hebras, orientados siempre 5'->3'."""
    normalizedSequence = Seq(str(dnaSequence).upper().replace("U", "T"))
    reverseSequence = normalizedSequence.reverse_complement()
    frames = []
    for strand, sequence in (("+", normalizedSequence), ("-", reverseSequence)):
        for frame in range(3):
            frames.append((strand, frame, sequence[frame:]))
    return frames


def findOrfsInFrame(frameSequence: Seq, strand: str, frame: int) -> list[Orf]:
    """Busca cada ORF desde ATG hasta el primer stop en el mismo marco."""
    orfs = []
    sequenceText = str(frameSequence)
    for start in range(0, len(sequenceText) - 2, 3):
        if sequenceText[start : start + 3] != START_CODON:
            continue
        for stopStart in range(start + 3, len(sequenceText) - 2, 3):
            if sequenceText[stopStart : stopStart + 3] not in STOP_CODONS:
                continue
            end = stopStart + 3
            dnaOrf = Seq(sequenceText[start:end])
            protein = dnaOrf[:-3].translate()
            orfs.append(Orf(strand, frame, start + frame, end + frame, dnaOrf, protein))
            break
    return orfs


def findAllOrfs(dnaSequence: Seq) -> list[Orf]:
    """Encuentra ORFs en los seis marcos de lectura posibles."""
    return [
        orf
        for strand, frame, frameSequence in getReadingFrames(dnaSequence)
        for orf in findOrfsInFrame(frameSequence, strand, frame)
    ]


def generateRandomDnaSequences(
    numberOfSequences: int, sequenceLength: int, seed: int = DEFAULT_SEED
) -> list[Seq]:
    """Genera ADN aleatorio uniforme; las cuatro bases tienen P = 0.25."""
    if numberOfSequences <= 0 or sequenceLength <= 0:
        raise ValueError("La cantidad y longitud de secuencias aleatorias deben ser positivas.")
    randomGenerator = random.Random(seed)
    return [
        Seq("".join(randomGenerator.choices(DNA_BASES, k=sequenceLength)))
        for _ in range(numberOfSequences)
    ]


def generateRandomOrfs(
    numberOfSequences: int, sequenceLength: int, seed: int = DEFAULT_SEED
) -> list[Orf]:
    """Genera ADN al azar y obtiene sus ORFs con el mismo detector usado en el gen."""
    randomOrfs = []
    for dnaSequence in generateRandomDnaSequences(numberOfSequences, sequenceLength, seed):
        randomOrfs.extend(findAllOrfs(dnaSequence))
    if not randomOrfs:
        raise ValueError("No se generaron ORFs aleatorios; aumente la longitud o las muestras.")
    return randomOrfs


def getAminoAcidCounts(proteinSequence: Seq) -> np.ndarray:
    """Cuenta los 20 aminoacidos estandar, ignorando caracteres ambiguos."""
    counts = Counter(aminoAcid for aminoAcid in str(proteinSequence) if aminoAcid in AMINO_ACIDS)
    return np.array([counts[aminoAcid] for aminoAcid in AMINO_ACIDS], dtype=float)


def buildLengthModel(
    codingSequences: list[Seq], randomSequences: list[Seq], smoothing: float = SMOOTHING
) -> LengthModel:
    """Estima P(L|C) y P(L|R) en bins comunes con smoothing de Laplace."""
    codingLengths = np.array([sum(getAminoAcidCounts(sequence)) for sequence in codingSequences])
    randomLengths = np.array([sum(getAminoAcidCounts(sequence)) for sequence in randomSequences])
    allLengths = np.concatenate((codingLengths, randomLengths))
    binEdges = np.histogram_bin_edges(allLengths, bins="fd")
    if len(binEdges) < 2:
        binEdges = np.array([0, max(1, int(allLengths[0])) + 1])

    codingCounts, _ = np.histogram(codingLengths, bins=binEdges)
    randomCounts, _ = np.histogram(randomLengths, bins=binEdges)
    codingProbabilities = (codingCounts + smoothing) / (len(codingLengths) + smoothing * len(codingCounts))
    randomProbabilities = (randomCounts + smoothing) / (len(randomLengths) + smoothing * len(randomCounts))
    return LengthModel(binEdges, codingProbabilities, randomProbabilities)


def trainOrfClassifier(
    realProteins: list[SeqRecord], randomOrfs: list[Orf], smoothing: float = SMOOTHING
) -> OrfClassifier:
    """Ajusta el modelo de longitudes y el multinomial de aminoacidos para C y R."""
    codingSequences = [protein.seq for protein in realProteins]
    randomSequences = [orf.proteinSequence for orf in randomOrfs]
    lengthModel = buildLengthModel(codingSequences, randomSequences, smoothing)

    codingCounts = np.sum([getAminoAcidCounts(sequence) for sequence in codingSequences], axis=0)
    randomCounts = np.sum([getAminoAcidCounts(sequence) for sequence in randomSequences], axis=0)
    codingProbabilities = (codingCounts + smoothing) / (codingCounts.sum() + smoothing * len(AMINO_ACIDS))
    randomProbabilities = (randomCounts + smoothing) / (randomCounts.sum() + smoothing * len(AMINO_ACIDS))
    return OrfClassifier(lengthModel, codingProbabilities, randomProbabilities)


def plotClassifierResults(
    classifier: OrfClassifier,
    realProteins: list[SeqRecord],
    randomOrfs: list[Orf],
    targetOrfs: list[Orf],
    outputPath: Path = Path("clasificador_orfs.png"),
) -> None:
    """Grafica las distribuciones aprendidas y los posteriores de los ORFs objetivo."""
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    figure.patch.set_facecolor("#f7f7f5")
    codingColor = "#2a6f97"
    randomColor = "#e07a5f"
    targetColor = "#6a4c93"

    axes[0, 0].stairs(classifier.lengthModel.codingProbabilities,
                      classifier.lengthModel.binEdges, color=codingColor, linewidth=2.2,
                      label="C: proteinas reales")
    axes[0, 0].stairs(classifier.lengthModel.randomProbabilities,
                      classifier.lengthModel.binEdges, color=randomColor, linewidth=2.2,
                      label="R: ORFs de ADN aleatorio")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Longitud (aminoacidos, escala log)")
    axes[0, 0].set_ylabel("Probabilidad del bin")
    axes[0, 0].set_title("P(L | C) frente a P(L | R), con smoothing", fontweight="bold")
    axes[0, 0].legend(frameon=False)

    codingOrder = np.argsort(classifier.codingAminoAcidProbabilities)[::-1]
    positions = np.arange(len(AMINO_ACIDS))
    labels = [AMINO_ACIDS[index] for index in codingOrder]
    axes[0, 1].bar(positions - 0.2, classifier.codingAminoAcidProbabilities[codingOrder],
                   width=0.4, color=codingColor, label="C: reales")
    axes[0, 1].bar(positions + 0.2, classifier.randomAminoAcidProbabilities[codingOrder],
                   width=0.4, color=randomColor, label="R: aleatorios")
    axes[0, 1].set_xticks(positions, labels)
    axes[0, 1].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axes[0, 1].set_xlabel("Aminoacido (ordenado por frecuencia en C)")
    axes[0, 1].set_ylabel("Frecuencia estimada")
    axes[0, 1].set_title("Modelo multinomial de aminoacidos", fontweight="bold")
    axes[0, 1].legend(frameon=False)

    targetLengths = [orf.aminoAcidLength for orf in targetOrfs]
    targetProbabilities = [classifier.codingProbability(orf.proteinSequence) for orf in targetOrfs]
    if targetOrfs:
        axes[1, 0].scatter(targetLengths, targetProbabilities, s=32, alpha=0.76,
                           color=targetColor, edgecolor="white", linewidth=0.4)
    axes[1, 0].axhline(0.5, color="#243447", linestyle="--", linewidth=1,
                       label="Umbral P(C|datos) = 0.5")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_ylim(-0.03, 1.03)
    axes[1, 0].set_xlabel("Longitud del ORF (aminoacidos, escala log)")
    axes[1, 0].set_ylabel("P(C | L, composicion)")
    axes[1, 0].set_title("Probabilidad de codificar por ORF", fontweight="bold")
    axes[1, 0].legend(frameon=False)

    if targetOrfs:
        axes[1, 1].hist(targetProbabilities, bins=np.linspace(0, 1, 21), color=targetColor,
                        alpha=0.85, edgecolor="white")
    axes[1, 1].axvline(0.5, color="#243447", linestyle="--", linewidth=1)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_xlabel("P(C | L, composicion)")
    axes[1, 1].set_ylabel("Cantidad de ORFs")
    axes[1, 1].set_title("Distribucion de probabilidades posteriores", fontweight="bold")

    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)

    figure.suptitle("Clasificador bayesiano generativo de ORFs", fontsize=18,
                    fontweight="bold", color="#243447")
    figure.tight_layout()
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(outputPath, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def printOrfSummary(classifier: OrfClassifier, orfs: list[Orf]) -> None:
    """Muestra ORFs ordenados por probabilidad posterior descendente."""
    print(f"ORFs detectados: {len(orfs)}")
    print("hebra\tmarco\tinicio\tfin\tlongitud (aa)\tP(codificante)")
    scoredOrfs = [
        (orf, classifier.codingProbability(orf.proteinSequence)) for orf in orfs
    ]
    for orf, probability in sorted(scoredOrfs, key=lambda item: item[1], reverse=True):
        print(
            f"{orf.strand}\t{orf.frame}\t{orf.start}\t{orf.end}\t"
            f"{orf.aminoAcidLength}\t{probability:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True, help="Email requerido por NCBI Entrez")
    parser.add_argument("--gen", help="Simbolo de gen humano, por ejemplo BRCA1")
    parser.add_argument("--proteinas", type=int, default=1000,
                        help="Cantidad de proteinas humanas para entrenar C")
    parser.add_argument("--muestras-azar", type=int, default=10,
                        help="Cantidad de secuencias de ADN aleatorio para entrenar R")
    parser.add_argument("--longitud-azar", type=int, default=100_000,
                        help="Longitud en nucleotidos de cada secuencia aleatoria")
    parser.add_argument("--semilla", type=int, default=DEFAULT_SEED)
    parser.add_argument("--salida", type=Path, default=Path("clasificador_orfs.png"))
    arguments = parser.parse_args()

    geneRecord = downloadHumanGene(arguments.email, arguments.gen, arguments.semilla)
    targetOrfs = findAllOrfs(geneRecord.seq)
    realProteins = downloadHumanProteins(
        arguments.email, arguments.proteinas, seed=arguments.semilla
    )
    randomOrfs = generateRandomOrfs(
        arguments.muestras_azar, arguments.longitud_azar, arguments.semilla
    )
    classifier = trainOrfClassifier(realProteins, randomOrfs)

    print(f"Secuencia objetivo: {geneRecord.id}")
    print(f"Descripcion: {geneRecord.description}")
    print(f"Longitud genomica: {len(geneRecord.seq)} nt")
    print(f"Proteinas reales para C: {len(realProteins)}")
    print(f"ORFs aleatorios para R: {len(randomOrfs)}")
    printOrfSummary(classifier, targetOrfs)
    plotClassifierResults(classifier, realProteins, randomOrfs, targetOrfs, arguments.salida)
    print(f"Grafico guardado en: {arguments.salida.resolve()}")


if __name__ == "__main__":
    main()
