"""Exploración de regiones codificantes con ventanas y uso de codones.

El programa aprende un clasificador bayesiano desde CDS humanos reales y ADN
aleatorio, y asigna a cada ventana un posterior de pertenecer a una región
codificante usando exclusivamente su distribución de codones.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord


CODON_BASES = "TCAG"
STOP_CODONS = {"TAA", "TAG", "TGA"}
DEFAULT_WINDOW_LENGTHS = (90, 150, 300, 600, 900)


@dataclass
class DnaWindow:
    """Ventana de ADN con coordenadas 0-based y extremo final excluyente."""

    start: int
    end: int
    sequence: Seq


@dataclass
class CdsSegment:
    """Una parte genómica de un CDS, con coordenadas 0-based semiabiertas."""

    cdsIndex: int
    partIndex: int
    start: int
    end: int
    strand: int | None
    sequence: Seq


@dataclass
class CdsRegion:
    """CDS anotado completo y sus partes, si está formado por varios segmentos."""

    cdsIndex: int
    location: str
    codonStart: int
    sequence: Seq
    segments: list[CdsSegment]


@dataclass
class CdsWindow:
    """Ventana dentro de un segmento CDS y sus coordenadas genómicas."""

    cdsIndex: int
    partIndex: int
    genomicStart: int
    genomicEnd: int
    strand: int | None
    sequence: Seq


@dataclass
class WindowSizeResult:
    """Desempeño del clasificador para un tamaño de ventana."""

    windowLength: int
    positiveWindows: int
    positiveRate: float
    falsePositiveRate: float

    @property
    def specificity(self) -> float:
        return 100 - self.falsePositiveRate


def downloadHumanGenomicRecords(email: str, geneNames: list[str]) -> list[SeqRecord]:
    """Descarga registros RefSeqGene humanos en formato GenBank con annotations."""
    if not geneNames:
        raise ValueError("Se debe indicar al menos un gen humano.")

    Entrez.email = email
    records = []
    for geneName in geneNames:
        searchQuery = (
            f'Homo sapiens[Organism] AND refseqgene[filter] '
            f'AND "{geneName}"[Gene Name]'
        )
        with Entrez.esearch(db="nuccore", term=searchQuery, retmax=20) as searchHandle:
            nucleotideIds = Entrez.read(searchHandle)["IdList"]
        if not nucleotideIds:
            raise ValueError(f"No se encontró una entrada RefSeqGene para {geneName}.")

        with Entrez.efetch(
            db="nuccore", id=nucleotideIds[0], rettype="gb", retmode="text"
        ) as recordHandle:
            records.append(SeqIO.read(recordHandle, "genbank"))
        sleep(0.34)
    return records


def downloadRandomHumanGenomicRecords(
    email: str, numberOfGenes: int = 100, seed: int = 42
) -> list[SeqRecord]:
    """Descarga una muestra aleatoria de registros humanos RefSeqGene en GenBank.

    Los identificadores se mezclan dentro de un conjunto de hasta 10.000 resultados.
    Los registros sin CDS se descartan y se reemplazan automáticamente hasta
    completar la cantidad solicitada.
    """
    if numberOfGenes <= 0:
        raise ValueError("La cantidad de genes debe ser mayor que cero.")

    Entrez.email = email
    searchQuery = "Homo sapiens[Organism] AND refseqgene[filter]"
    with Entrez.esearch(db="nuccore", term=searchQuery, retmax=10_000) as searchHandle:
        nucleotideIds = Entrez.read(searchHandle)["IdList"]
    if len(nucleotideIds) < numberOfGenes:
        raise ValueError(
            f"NCBI devolvió {len(nucleotideIds)} registros RefSeqGene; "
            f"se solicitaron {numberOfGenes}."
        )

    randomGenerator = random.Random(seed)
    randomGenerator.shuffle(nucleotideIds)
    recordsWithCds = []
    discardedRecords = 0

    for batchStart in range(0, len(nucleotideIds), 10):
        batchIds = nucleotideIds[batchStart : batchStart + 10]
        with Entrez.efetch(
            db="nuccore", id=",".join(batchIds), rettype="gb", retmode="text"
        ) as recordHandle:
            batchRecords = list(SeqIO.parse(recordHandle, "genbank"))
        sleep(0.34)

        for record in batchRecords:
            if any(feature.type == "CDS" for feature in record.features):
                recordsWithCds.append(record)
            else:
                discardedRecords += 1
        if len(recordsWithCds) >= numberOfGenes:
            if discardedRecords:
                print(f"Registros sin CDS descartados: {discardedRecords}")
            return recordsWithCds[:numberOfGenes]

    raise ValueError(
        f"Sólo se encontraron {len(recordsWithCds)} registros con CDS anotado; "
        f"se solicitaron {numberOfGenes}."
    )


def slidingDnaWindows(
    dnaSequence: Seq, windowLength: int, step: int = 1
) -> Iterator[DnaWindow]:
    """Recorre ADN con ventanas completas de ``windowLength`` nucleótidos.

    No devuelve una ventana incompleta al final. El parámetro ``step`` permite
    desplazarla de a uno o de a varios nucleótidos.
    """
    if windowLength <= 0:
        raise ValueError("La longitud de ventana debe ser mayor que cero.")
    if step <= 0:
        raise ValueError("El paso debe ser mayor que cero.")

    for start in range(0, len(dnaSequence) - windowLength + 1, step):
        end = start + windowLength
        yield DnaWindow(start, end, dnaSequence[start:end])


def countWindows(sequenceLength: int, windowLength: int, step: int) -> int:
    """Calcula cuántas ventanas completas producirá ``slidingDnaWindows``."""
    if windowLength <= 0:
        raise ValueError("La longitud de ventana debe ser mayor que cero.")
    if step <= 0:
        raise ValueError("El paso debe ser mayor que cero.")
    if sequenceLength < windowLength:
        return 0
    return 1 + (sequenceLength - windowLength) // step


def extractCdsRegions(genomicRecord: SeqRecord) -> list[CdsRegion]:
    """Extrae exclusivamente features CDS y conserva las partes de cada location.

    La secuencia de ``CdsRegion`` corresponde al CDS completo, empalmado según la
    annotation. Cada ``CdsSegment`` conserva una parte genómica individual para
    generar ventanas que no atraviesen intrones. No se usan features ``exon``, ya
    que pueden contener UTRs no codificantes.
    """
    codingFeatures: list[SeqFeature] = [
        feature for feature in genomicRecord.features if feature.type == "CDS"
    ]
    cdsRegions = []

    for cdsIndex, feature in enumerate(codingFeatures, start=1):
        if feature.location is None:
            continue
        codonStart = int(feature.qualifiers.get("codon_start", ["1"])[0])
        partLocations = list(getattr(feature.location, "parts", [feature.location]))
        segments = []
        for partIndex, part in enumerate(partLocations, start=1):
            segments.append(
                CdsSegment(
                    cdsIndex=cdsIndex,
                    partIndex=partIndex,
                    start=int(part.start),
                    end=int(part.end),
                    strand=part.strand,
                    sequence=part.extract(genomicRecord.seq),
                )
            )
        cdsRegions.append(
            CdsRegion(
                cdsIndex=cdsIndex,
                location=str(feature.location),
                codonStart=codonStart,
                sequence=feature.extract(genomicRecord.seq),
                segments=segments,
            )
        )
    return cdsRegions


def slidingCdsWindows(
    cdsSegment: CdsSegment, windowLength: int, step: int = 1
) -> Iterator[CdsWindow]:
    """Genera ventanas en un segmento CDS sin cruzar límites de location.part.

    La secuencia se orienta 5'->3' respecto del CDS. Para segmentos de hebra
    negativa, las coordenadas genómicas se convierten a la orientación original.
    """
    for window in slidingDnaWindows(cdsSegment.sequence, windowLength, step):
        if cdsSegment.strand == -1:
            genomicStart = cdsSegment.end - window.end
            genomicEnd = cdsSegment.end - window.start
        else:
            genomicStart = cdsSegment.start + window.start
            genomicEnd = cdsSegment.start + window.end
        yield CdsWindow(
            cdsIndex=cdsSegment.cdsIndex,
            partIndex=cdsSegment.partIndex,
            genomicStart=genomicStart,
            genomicEnd=genomicEnd,
            strand=cdsSegment.strand,
            sequence=window.sequence,
        )


def getCdsCodonCounts(
    genomicRecords: list[SeqRecord], includeStopCodons: bool = False
) -> Counter[str]:
    """Cuenta codones en CDS empalmados, respetando el qualifier ``codon_start``.

    Cada CDS se recorre como una secuencia completa, no como partes independientes,
    para no desalinear codones que atraviesan un límite de exón. Por defecto se
    excluyen los codones stop, porque el uso de codones suele definirse sobre los
    61 codones con sentido.
    """
    codonCounts: Counter[str] = Counter()
    for record in genomicRecords:
        for cdsRegion in extractCdsRegions(record):
            codingSequence = str(cdsRegion.sequence)[cdsRegion.codonStart - 1 :].upper()
            for start in range(0, len(codingSequence) - 2, 3):
                codon = codingSequence[start : start + 3]
                if set(codon) - set("ACGT"):
                    continue
                if not includeStopCodons and codon in STOP_CODONS:
                    continue
                codonCounts[codon] += 1
    return codonCounts


def getCodonOrder() -> list[str]:
    """Devuelve los 64 codones en el orden convencional de la tabla genética."""
    return [
        firstBase + secondBase + thirdBase
        for firstBase in CODON_BASES
        for secondBase in CODON_BASES
        for thirdBase in CODON_BASES
    ]


@dataclass
class CodonBayesianClassifier:
    """Clasificador multinomial basado únicamente en frecuencias de codones."""

    codingProbabilities: dict[str, float]
    randomProbabilities: dict[str, float]
    codingPrior: float = 0.5
    randomPrior: float = 0.5

    def logLikelihoodRatio(self, codonCounts: Counter[str]) -> float:
        """Calcula log prior odds más la suma de log odds por codón."""
        score = math.log(self.codingPrior / self.randomPrior)
        for codon in getCodonOrder():
            score += codonCounts[codon] * math.log(
                self.codingProbabilities[codon] / self.randomProbabilities[codon]
            )
        return score

    def codingProbability(self, codonCounts: Counter[str]) -> float:
        """Convierte el log-likelihood ratio en una probabilidad posterior."""
        score = self.logLikelihoodRatio(codonCounts)
        if score >= 0:
            return 1 / (1 + math.exp(-score))
        exponentialScore = math.exp(score)
        return exponentialScore / (1 + exponentialScore)


def generateRandomCodonCounts(numberOfCodons: int, seed: int) -> Counter[str]:
    """Genera una distribución empírica de codones bajo ADN uniforme al azar."""
    if numberOfCodons <= 0:
        raise ValueError("La cantidad de codones aleatorios debe ser positiva.")
    codons = getCodonOrder()
    randomGenerator = np.random.default_rng(seed)
    counts = randomGenerator.multinomial(numberOfCodons, [1 / len(codons)] * len(codons))
    return Counter({codon: int(count) for codon, count in zip(codons, counts)})


def trainCodonClassifier(
    codingCounts: Counter[str], randomCounts: Counter[str], smoothing: float = 1.0
) -> CodonBayesianClassifier:
    """Estima P(codón|C) y P(codón|R) con smoothing de Laplace."""
    if smoothing <= 0:
        raise ValueError("El smoothing debe ser mayor que cero.")
    codons = getCodonOrder()
    codingTotal = sum(codingCounts.values())
    randomTotal = sum(randomCounts.values())
    if codingTotal == 0 or randomTotal == 0:
        raise ValueError("Ambos conjuntos de entrenamiento deben contener codones.")

    codingProbabilities = {
        codon: (codingCounts[codon] + smoothing)
        / (codingTotal + smoothing * len(codons))
        for codon in codons
    }
    randomProbabilities = {
        codon: (randomCounts[codon] + smoothing)
        / (randomTotal + smoothing * len(codons))
        for codon in codons
    }
    return CodonBayesianClassifier(codingProbabilities, randomProbabilities)


def getFrameCodonCounts(dnaSequence: Seq, frame: int) -> Counter[str]:
    """Cuenta codones completos y no ambiguos en un marco específico."""
    sequenceText = str(dnaSequence).upper().replace("U", "T")
    counts: Counter[str] = Counter()
    for start in range(frame, len(sequenceText) - 2, 3):
        codon = sequenceText[start : start + 3]
        if not set(codon) - set("ACGT"):
            counts[codon] += 1
    return counts


def scoreDnaWindow(classifier: CodonBayesianClassifier, dnaSequence: Seq) -> float:
    """Devuelve el máximo posterior entre los seis marcos de una ventana."""
    normalizedSequence = Seq(str(dnaSequence).upper().replace("U", "T"))
    reverseSequence = normalizedSequence.reverse_complement()
    probabilities = [
        classifier.codingProbability(getFrameCodonCounts(sequence, frame))
        for sequence in (normalizedSequence, reverseSequence)
        for frame in range(3)
    ]
    return max(probabilities)


def iterateCdsWindows(
    genomicRecords: list[SeqRecord], windowLength: int, step: int
) -> Iterator[Seq]:
    """Genera ventanas positivas desde CDS empalmados y alineados por codon_start."""
    for record in genomicRecords:
        for cdsRegion in extractCdsRegions(record):
            alignedSequence = cdsRegion.sequence[cdsRegion.codonStart - 1 :]
            for window in slidingDnaWindows(alignedSequence, windowLength, step):
                yield window.sequence


def sampleSequences(
    sequences: Iterator[Seq], maximumSequences: int, seed: int
) -> list[Seq]:
    """Toma una muestra uniforme con reservoir sampling sin guardar todas las ventanas."""
    if maximumSequences <= 0:
        raise ValueError("La cantidad máxima de ventanas debe ser positiva.")
    randomGenerator = random.Random(seed)
    sample: list[Seq] = []
    for index, sequence in enumerate(sequences):
        if index < maximumSequences:
            sample.append(sequence)
            continue
        replacementIndex = randomGenerator.randint(0, index)
        if replacementIndex < maximumSequences:
            sample[replacementIndex] = sequence
    return sample


def getPositiveControlScores(
    classifier: CodonBayesianClassifier,
    controlRecords: list[SeqRecord],
    windowLength: int,
    step: int,
    maximumWindows: int,
    seed: int,
) -> list[float]:
    """Puntúa ventanas de CDS de genes que no participaron del entrenamiento."""
    windows = sampleSequences(
        iterateCdsWindows(controlRecords, windowLength, step), maximumWindows, seed
    )
    if not windows:
        raise ValueError("Los genes de control no produjeron ventanas CDS completas.")
    return [scoreDnaWindow(classifier, window) for window in windows]


def getNegativeControlScores(
    classifier: CodonBayesianClassifier,
    numberOfWindows: int,
    windowLength: int,
    seed: int,
) -> list[float]:
    """Puntúa ventanas nuevas de ADN uniforme que no se usaron para entrenar."""
    randomGenerator = random.Random(seed)
    return [
        scoreDnaWindow(
            classifier,
            Seq("".join(randomGenerator.choices("ACGT", k=windowLength))),
        )
        for _ in range(numberOfWindows)
    ]


def evaluateWindowSizes(
    classifier: CodonBayesianClassifier,
    controlRecords: list[SeqRecord],
    windowLengths: list[int],
    step: int,
    maximumWindows: int,
    threshold: float,
    seed: int,
) -> list[WindowSizeResult]:
    """Evalúa sensibilidad y falsos positivos para varios tamaños de ventana."""
    if len(windowLengths) != 5:
        raise ValueError("La comparación requiere exactamente cinco tamaños de ventana.")
    if any(length <= 0 or length % 3 != 0 for length in windowLengths):
        raise ValueError("Los tamaños de ventana deben ser positivos y múltiplos de tres.")
    if len(set(windowLengths)) != len(windowLengths):
        raise ValueError("Los cinco tamaños de ventana deben ser diferentes.")

    results = []
    for index, windowLength in enumerate(sorted(windowLengths)):
        positiveScores = getPositiveControlScores(
            classifier,
            controlRecords,
            windowLength,
            step,
            maximumWindows,
            seed + index * 2,
        )
        negativeScores = getNegativeControlScores(
            classifier,
            len(positiveScores),
            windowLength,
            seed + index * 2 + 1,
        )
        positiveRate = (
            100 * sum(score > threshold for score in positiveScores) / len(positiveScores)
        )
        falsePositiveRate = (
            100 * sum(score > threshold for score in negativeScores) / len(negativeScores)
        )
        results.append(
            WindowSizeResult(
                windowLength,
                len(positiveScores),
                positiveRate,
                falsePositiveRate,
            )
        )
    return results


def plotWindowSizeResults(
    results: list[WindowSizeResult],
    threshold: float,
    outputPath: Path = Path("comparacion_tamanos_ventana.png"),
) -> None:
    """Grafica cómo cambia el control positivo y negativo con el tamaño."""
    windowLengths = [result.windowLength for result in results]
    positiveRates = [result.positiveRate for result in results]
    falsePositiveRates = [result.falsePositiveRate for result in results]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    figure.patch.set_facecolor("#f7f7f5")
    codingColor = "#2a6f97"
    randomColor = "#d1495b"

    axis.plot(
        windowLengths, positiveRates, marker="o", markersize=8, linewidth=2.5,
        color=codingColor, label="Aciertos en CDS reales",
    )
    axis.plot(
        windowLengths, falsePositiveRates, marker="o", markersize=8, linewidth=2.5,
        color=randomColor, label="Falsos positivos en ADN aleatorio",
    )
    for windowLength, positiveRate, falsePositiveRate in zip(
        windowLengths, positiveRates, falsePositiveRates
    ):
        axis.annotate(
            f"{positiveRate:.1f}%", (windowLength, positiveRate),
            xytext=(0, 10), textcoords="offset points", ha="center",
            color=codingColor, fontweight="bold",
        )
        axis.annotate(
            f"{falsePositiveRate:.1f}%", (windowLength, falsePositiveRate),
            xytext=(0, -17), textcoords="offset points", ha="center",
            color=randomColor, fontweight="bold",
        )

    axis.set_xticks(windowLengths)
    axis.set_ylim(-3, 103)
    axis.set_xlabel("Tamaño de ventana (nucleótidos)")
    axis.set_ylabel("Ventanas clasificadas como codificantes (%)")
    axis.set_title(
        f"Efecto del tamaño de ventana (umbral P(C) > {threshold:g})",
        fontsize=15, fontweight="bold", color="#243447",
    )
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.2)
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(outputPath, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def saveWindowSizeResults(results: list[WindowSizeResult], outputPath: Path) -> None:
    """Guarda los resultados de los cinco tamaños en una tabla TSV."""
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    with outputPath.open("w", newline="", encoding="utf-8") as outputFile:
        writer = csv.writer(outputFile, delimiter="\t")
        writer.writerow([
            "ventana_nt", "ventanas_positivas", "aciertos_positivos_pct",
            "falsos_positivos_pct", "especificidad_pct",
        ])
        for result in results:
            writer.writerow([
                result.windowLength,
                result.positiveWindows,
                result.positiveRate,
                result.falsePositiveRate,
                result.specificity,
            ])


def plotCodonClassifierControls(
    classifier: CodonBayesianClassifier,
    positiveScores: list[float],
    negativeScores: list[float],
    threshold: float,
    outputPath: Path = Path("control_clasificador_codones.png"),
) -> None:
    """Grafica los log odds, posteriores y desempeño de ambos controles."""
    positiveRate = 100 * sum(score > threshold for score in positiveScores) / len(positiveScores)
    falsePositiveRate = 100 * sum(score > threshold for score in negativeScores) / len(negativeScores)
    codonLogOdds = {
        codon: math.log2(
            classifier.codingProbabilities[codon] / classifier.randomProbabilities[codon]
        )
        for codon in getCodonOrder()
    }
    orderedCodons = sorted(codonLogOdds, key=codonLogOdds.get, reverse=True)
    logOddsValues = [codonLogOdds[codon] for codon in orderedCodons]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=[1, 1.15])
    scoreAxis = figure.add_subplot(grid[0, 0])
    rateAxis = figure.add_subplot(grid[0, 1])
    oddsAxis = figure.add_subplot(grid[1, :])
    figure.patch.set_facecolor("#f7f7f5")
    codingColor = "#2a6f97"
    randomColor = "#d1495b"

    bins = np.linspace(0, 1, 31)
    scoreAxis.hist(positiveScores, bins=bins, density=True, alpha=0.72,
                   color=codingColor, label="CDS reales reservados")
    scoreAxis.hist(negativeScores, bins=bins, density=True, alpha=0.62,
                   color=randomColor, label="ADN aleatorio")
    scoreAxis.axvline(threshold, color="#243447", linestyle="--",
                      label=f"Umbral: {threshold:g}")
    scoreAxis.set_xlabel("P(C | distribución de codones)")
    scoreAxis.set_ylabel("Densidad")
    scoreAxis.set_title("Distribución de scores", fontweight="bold")
    scoreAxis.legend(frameon=False)

    bars = rateAxis.bar(
        ["Control positivo\nCDS reales", "Falsos positivos\nADN aleatorio"],
        [positiveRate, falsePositiveRate], color=[codingColor, randomColor], width=0.62,
    )
    rateAxis.set_ylim(0, 105)
    rateAxis.set_ylabel("Ventanas clasificadas como codificantes (%)")
    rateAxis.set_title("Rendimiento con datos no vistos", fontweight="bold")
    rateAxis.bar_label(
        bars, labels=[f"{positiveRate:.1f}%", f"{falsePositiveRate:.1f}%"],
        padding=5, fontsize=11, fontweight="bold",
    )

    colors = [codingColor if value >= 0 else randomColor for value in logOddsValues]
    oddsAxis.bar(range(len(orderedCodons)), logOddsValues, color=colors, width=0.82)
    oddsAxis.axhline(0, color="#243447", linewidth=1)
    oddsAxis.set_xticks(range(len(orderedCodons)), orderedCodons, rotation=90, fontsize=8)
    oddsAxis.set_ylabel(r"Contribución por codón: $\log_2(p_c^C/p_c^R)$")
    oddsAxis.set_title("Codones que favorecen C (azul) o R (rojo)", fontweight="bold")

    for axis in (scoreAxis, rateAxis, oddsAxis):
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
    figure.suptitle("Clasificador bayesiano basado sólo en uso de codones",
                    fontsize=18, fontweight="bold", color="#243447")
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(outputPath, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def plotCdsCodonDistribution(
    codonCounts: Counter[str], numberOfGenes: int,
    outputPath: Path = Path("distribucion_codones_cds.png"),
) -> None:
    """Grafica el uso de codones en cuatro mapas de calor, uno por primera base."""
    totalCodons = sum(codonCounts.values())
    if totalCodons == 0:
        raise ValueError("No se encontraron codones válidos en los CDS descargados.")

    frequencies = {codon: count / totalCodons for codon, count in codonCounts.items()}
    maximumFrequency = max(frequencies.values())
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    figure.patch.set_facecolor("#f7f7f5")
    image = None

    for axis, firstBase in zip(axes.flat, CODON_BASES):
        values = np.zeros((4, 4))
        for thirdIndex, thirdBase in enumerate(CODON_BASES):
            for secondIndex, secondBase in enumerate(CODON_BASES):
                codon = firstBase + secondBase + thirdBase
                values[thirdIndex, secondIndex] = frequencies.get(codon, 0)

        image = axis.imshow(values, cmap="YlGnBu", vmin=0, vmax=maximumFrequency)
        axis.set_xticks(range(4), CODON_BASES)
        axis.set_yticks(range(4), CODON_BASES)
        axis.set_xlabel("Segunda base")
        axis.set_ylabel("Tercera base")
        axis.set_title(f"Primera base: {firstBase}", fontweight="bold")

        for thirdIndex, thirdBase in enumerate(CODON_BASES):
            for secondIndex, secondBase in enumerate(CODON_BASES):
                codon = firstBase + secondBase + thirdBase
                frequency = frequencies.get(codon, 0)
                label = "STOP" if codon in STOP_CODONS and codon not in codonCounts else codon
                textColor = "white" if frequency > maximumFrequency * 0.55 else "#243447"
                axis.text(
                    secondIndex, thirdIndex, f"{label}\n{frequency:.2%}",
                    ha="center", va="center", fontsize=9, color=textColor,
                )
        axis.grid(False)

    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82)
    colorbar.set_label("Frecuencia relativa")
    figure.suptitle(
        f"Uso de codones en CDS de {numberOfGenes} genes humanos aleatorios\n"
        f"{totalCodons:,} codones analizados",
        fontsize=16, fontweight="bold", color="#243447",
    )
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(outputPath, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def saveCodonDistribution(codonCounts: Counter[str], outputPath: Path) -> None:
    """Guarda counts y frecuencias para reutilizarlos al construir un score."""
    totalCodons = sum(codonCounts.values())
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    with outputPath.open("w", newline="", encoding="utf-8") as outputFile:
        writer = csv.writer(outputFile, delimiter="\t")
        writer.writerow(["codon", "count", "frecuencia_relativa"])
        for codon in getCodonOrder():
            count = codonCounts[codon]
            writer.writerow([codon, count, count / totalCodons if totalCodons else 0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True, help="Email requerido por NCBI Entrez")
    parser.add_argument(
        "--genes", nargs="+",
        help="Genes opcionales para explorar ventanas, por ejemplo HBB INS TP53",
    )
    parser.add_argument(
        "--ventana", type=int, default=300,
        help="Longitud de la ventana deslizante en nucleótidos",
    )
    parser.add_argument(
        "--paso", type=int, default=1,
        help="Desplazamiento de la ventana en nucleótidos",
    )
    parser.add_argument(
        "--genes-azar", type=int, default=100,
        help="Cantidad de genes RefSeqGene aleatorios para la distribución de codones",
    )
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument(
        "--incluir-stops", action="store_true",
        help="Incluye los codones TAA, TAG y TGA en la distribución",
    )
    parser.add_argument(
        "--salida-codones", type=Path, default=Path("distribucion_codones_cds.png"),
    )
    parser.add_argument(
        "--tabla-codones", type=Path, default=Path("distribucion_codones_cds.tsv"),
    )
    parser.add_argument(
        "--umbral", type=float, default=0.5,
        help="Umbral posterior para clasificar una ventana como codificante",
    )
    parser.add_argument(
        "--max-ventanas-control", type=int, default=5000,
        help="Máximo de ventanas CDS reservadas para el control positivo",
    )
    parser.add_argument(
        "--salida-control", type=Path, default=Path("control_clasificador_codones.png"),
    )
    parser.add_argument(
        "--ventanas-comparacion", nargs=5, type=int,
        default=list(DEFAULT_WINDOW_LENGTHS),
        metavar="N",
        help="Cinco tamaños de ventana, positivos y múltiplos de tres",
    )
    parser.add_argument(
        "--salida-ventanas", type=Path, default=Path("comparacion_tamanos_ventana.png"),
    )
    parser.add_argument(
        "--tabla-ventanas", type=Path, default=Path("resultados_tamanos_ventana.tsv"),
    )
    arguments = parser.parse_args()
    if not 0 < arguments.umbral < 1:
        raise ValueError("El umbral debe estar estrictamente entre 0 y 1.")
    if arguments.genes_azar < 2:
        raise ValueError("Se necesitan al menos dos genes para separar entrenamiento y control.")

    randomRecords = downloadRandomHumanGenomicRecords(
        arguments.email, arguments.genes_azar, arguments.semilla
    )
    codonCounts = getCdsCodonCounts(randomRecords, arguments.incluir_stops)
    plotCdsCodonDistribution(codonCounts, len(randomRecords), arguments.salida_codones)
    saveCodonDistribution(codonCounts, arguments.tabla_codones)
    print(f"Genes aleatorios descargados: {len(randomRecords)}")
    print(f"Codones analizados en CDS: {sum(codonCounts.values())}")
    print(f"Gráfico guardado en: {arguments.salida_codones.resolve()}")
    print(f"Tabla guardada en: {arguments.tabla_codones.resolve()}")

    shuffledRecords = randomRecords.copy()
    random.Random(arguments.semilla + 1).shuffle(shuffledRecords)
    trainingCount = min(len(shuffledRecords) - 1, max(1, int(0.8 * len(shuffledRecords))))
    trainingRecords = shuffledRecords[:trainingCount]
    controlRecords = shuffledRecords[trainingCount:]

    codingTrainingCounts = getCdsCodonCounts(trainingRecords, includeStopCodons=True)
    randomTrainingCounts = generateRandomCodonCounts(
        sum(codingTrainingCounts.values()), arguments.semilla + 2
    )
    classifier = trainCodonClassifier(codingTrainingCounts, randomTrainingCounts)
    positiveScores = getPositiveControlScores(
        classifier,
        controlRecords,
        arguments.ventana,
        arguments.paso,
        arguments.max_ventanas_control,
        arguments.semilla + 3,
    )
    negativeScores = getNegativeControlScores(
        classifier, len(positiveScores), arguments.ventana, arguments.semilla + 4
    )
    plotCodonClassifierControls(
        classifier, positiveScores, negativeScores, arguments.umbral,
        arguments.salida_control,
    )
    positiveRate = 100 * sum(score > arguments.umbral for score in positiveScores) / len(positiveScores)
    falsePositiveRate = 100 * sum(score > arguments.umbral for score in negativeScores) / len(negativeScores)
    print(f"Genes de entrenamiento: {len(trainingRecords)}")
    print(f"Genes reservados para control positivo: {len(controlRecords)}")
    print(f"Ventanas positivas evaluadas: {len(positiveScores)}")
    print(f"Control positivo: {positiveRate:.1f}%")
    print(f"Falsos positivos: {falsePositiveRate:.1f}%")
    print(f"Gráfico de controles guardado en: {arguments.salida_control.resolve()}")

    windowSizeResults = evaluateWindowSizes(
        classifier,
        controlRecords,
        arguments.ventanas_comparacion,
        arguments.paso,
        arguments.max_ventanas_control,
        arguments.umbral,
        arguments.semilla + 10,
    )
    plotWindowSizeResults(
        windowSizeResults, arguments.umbral, arguments.salida_ventanas
    )
    saveWindowSizeResults(windowSizeResults, arguments.tabla_ventanas)
    print("Comparación de tamaños de ventana:")
    for result in windowSizeResults:
        print(
            f"  {result.windowLength} nt: aciertos {result.positiveRate:.1f}%, "
            f"falsos positivos {result.falsePositiveRate:.1f}%"
        )
    print(f"Gráfico de tamaños guardado en: {arguments.salida_ventanas.resolve()}")
    print(f"Tabla de tamaños guardada en: {arguments.tabla_ventanas.resolve()}")

    if not arguments.genes:
        return

    dnaRecords = downloadHumanGenomicRecords(arguments.email, arguments.genes)
    for record in dnaRecords:
        windowCount = countWindows(len(record.seq), arguments.ventana, arguments.paso)
        cdsRegions = extractCdsRegions(record)
        cdsSegments = [
            segment for region in cdsRegions for segment in region.segments
        ]
        codingWindowCount = sum(
            countWindows(len(segment.sequence), arguments.ventana, arguments.paso)
            for segment in cdsSegments
        )
        print(f"Secuencia: {record.id}")
        print(f"Descripción: {record.description}")
        print(f"Longitud: {len(record.seq)} nt")
        print(f"CDS anotados: {len(cdsRegions)}")
        print(f"Segmentos codificantes (location.part): {len(cdsSegments)}")
        print(
            f"Ventanas completas de {arguments.ventana} nt, paso {arguments.paso}: "
            f"{windowCount}"
        )
        print(f"Ventanas contenidas en segmentos CDS: {codingWindowCount}")


if __name__ == "__main__":
    main()
