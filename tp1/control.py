"""Controles positivo y negativo para el clasificador de ORFs.

El control positivo usa transcritos RefSeq mRNA de genes humanos conocidos.
Se usa mRNA y no ADN genomico porque un detector ATG->stop simple no hace
splicing y, por lo tanto, no puede reconstruir un CDS eucariota con intrones.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord

from distribucion import downloadHumanProteins
from orffind import (
    DEFAULT_SEED,
    Orf,
    OrfClassifier,
    findAllOrfs,
    generateRandomDnaSequences,
    generateRandomOrfs,
    trainOrfClassifier,
)


# Transcritos RefSeq humanos codificantes y bien caracterizados.
POSITIVE_CONTROL_TRANSCRIPTS = {
    "INS": "NM_000207.3",
    "HBB": "NM_000518.5",
    "TP53": "NM_000546.6",
    "BRCA1": "NM_007294.4",
    "CFTR": "NM_000492.4",
    "EGFR": "NM_005228.5",
    "MYC": "NM_002467.6",
    "ACTB": "NM_001101.5",
    "GAPDH": "NM_002046.7",
    "ALB": "NM_000477.7",
    "APOE": "NM_000041.4",
    "SOD1": "NM_000454.5",
    "TTR": "NM_000371.4",
    "LDHA": "NM_005566.4",
    "G6PD": "NM_000402.4",
}


@dataclass
class PositiveControlResult:
    geneName: str
    accession: str
    expectedLength: int
    detectedOrfs: int
    candidatesAboveThreshold: int
    expectedOrfRank: int | None

    @property
    def isHit(self) -> bool:
        return self.expectedOrfRank is not None


@dataclass
class NegativeControlResult:
    sequencesTested: int
    sequencesWithFalsePositive: int
    totalOrfs: int
    falsePositiveOrfs: int

    @property
    def falsePositiveSequenceRate(self) -> float:
        return self.sequencesWithFalsePositive / self.sequencesTested


def downloadAnnotatedTranscript(email: str, accession: str) -> SeqRecord:
    """Descarga un transcrito RefSeq junto con su anotacion GenBank."""
    Entrez.email = email
    with Entrez.efetch(
        db="nuccore", id=accession, rettype="gb", retmode="text"
    ) as recordHandle:
        transcriptRecord = SeqIO.read(recordHandle, "genbank")
    sleep(0.34)
    return transcriptRecord


def downloadGenomicGene(email: str, geneName: str) -> SeqRecord:
    """Descarga la entrada genomica RefSeqGene del gen humano indicado."""
    Entrez.email = email
    searchQuery = (
        f'Homo sapiens[Organism] AND refseqgene[filter] '
        f'AND "{geneName}"[Gene Name]'
    )
    with Entrez.esearch(db="nuccore", term=searchQuery, retmax=20) as searchHandle:
        nucleotideIds = Entrez.read(searchHandle)["IdList"]
    if not nucleotideIds:
        raise ValueError(f"No se encontro una entrada RefSeqGene para {geneName}.")

    with Entrez.efetch(
        db="nuccore", id=nucleotideIds[0], rettype="fasta", retmode="text"
    ) as recordHandle:
        genomicRecord = SeqIO.read(recordHandle, "fasta")
    sleep(0.34)
    return genomicRecord


def getAnnotatedProtein(transcriptRecord: SeqRecord) -> Seq:
    """Extrae la proteina del CDS anotado mas largo del transcrito."""
    codingFeatures: list[SeqFeature] = [
        feature
        for feature in transcriptRecord.features
        if feature.type == "CDS" and "translation" in feature.qualifiers
    ]
    if not codingFeatures:
        raise ValueError(f"El transcrito {transcriptRecord.id} no tiene un CDS anotado.")

    longestFeature = max(
        codingFeatures, key=lambda feature: len(feature.qualifiers["translation"][0])
    )
    return Seq(longestFeature.qualifiers["translation"][0].replace(" ", "").rstrip("*"))


def getTopCandidates(
    classifier: OrfClassifier, orfs: list[Orf], threshold: float, topCount: int
) -> list[tuple[Orf, float]]:
    """Filtra y ordena los candidatos que superan el umbral posterior."""
    candidates = [
        (orf, classifier.codingProbability(orf.proteinSequence))
        for orf in orfs
    ]
    candidates = [candidate for candidate in candidates if candidate[1] > threshold]
    return sorted(candidates, key=lambda candidate: candidate[1], reverse=True)[:topCount]


def runPositiveControls(
    email: str, classifier: OrfClassifier, threshold: float, topCount: int
) -> list[PositiveControlResult]:
    """Prueba si el CDS anotado aparece entre los mejores candidatos de cada gen."""
    results = []
    for geneName, accession in POSITIVE_CONTROL_TRANSCRIPTS.items():
        transcriptRecord = downloadAnnotatedTranscript(email, accession)
        expectedProtein = getAnnotatedProtein(transcriptRecord)
        detectedOrfs = findAllOrfs(transcriptRecord.seq)
        topCandidates = getTopCandidates(classifier, detectedOrfs, threshold, topCount)

        expectedOrfRank = next(
            (
                rank
                for rank, (orf, _) in enumerate(topCandidates, start=1)
                if str(orf.proteinSequence) == str(expectedProtein)
            ),
            None,
        )
        candidatesAboveThreshold = sum(
            classifier.codingProbability(orf.proteinSequence) > threshold
            for orf in detectedOrfs
        )
        results.append(
            PositiveControlResult(
                geneName,
                transcriptRecord.id,
                len(expectedProtein),
                len(detectedOrfs),
                candidatesAboveThreshold,
                expectedOrfRank,
            )
        )
    return results


def runGenomicControls(
    email: str, classifier: OrfClassifier, threshold: float, topCount: int
) -> list[PositiveControlResult]:
    """Busca los mismos CDS esperados, pero sobre ADN genomico sin splicing."""
    results = []
    for geneName, transcriptAccession in POSITIVE_CONTROL_TRANSCRIPTS.items():
        transcriptRecord = downloadAnnotatedTranscript(email, transcriptAccession)
        expectedProtein = getAnnotatedProtein(transcriptRecord)
        genomicRecord = downloadGenomicGene(email, geneName)
        detectedOrfs = findAllOrfs(genomicRecord.seq)
        topCandidates = getTopCandidates(classifier, detectedOrfs, threshold, topCount)

        expectedOrfRank = next(
            (
                rank
                for rank, (orf, _) in enumerate(topCandidates, start=1)
                if str(orf.proteinSequence) == str(expectedProtein)
            ),
            None,
        )
        candidatesAboveThreshold = sum(
            classifier.codingProbability(orf.proteinSequence) > threshold
            for orf in detectedOrfs
        )
        results.append(
            PositiveControlResult(
                geneName,
                genomicRecord.id,
                len(expectedProtein),
                len(detectedOrfs),
                candidatesAboveThreshold,
                expectedOrfRank,
            )
        )
    return results


def runNegativeControl(
    classifier: OrfClassifier,
    numberOfSequences: int,
    sequenceLength: int,
    threshold: float,
    seed: int,
) -> NegativeControlResult:
    """Mide cuantos ADN aleatorios producen al menos un ORF falso positivo."""
    sequencesWithFalsePositive = 0
    totalOrfs = 0
    falsePositiveOrfs = 0
    randomSequences = generateRandomDnaSequences(numberOfSequences, sequenceLength, seed)

    for sequence in randomSequences:
        detectedOrfs = findAllOrfs(sequence)
        probabilities = [
            classifier.codingProbability(orf.proteinSequence) for orf in detectedOrfs
        ]
        sequenceFalsePositives = sum(probability > threshold for probability in probabilities)
        totalOrfs += len(detectedOrfs)
        falsePositiveOrfs += sequenceFalsePositives
        if sequenceFalsePositives > 0:
            sequencesWithFalsePositive += 1

    return NegativeControlResult(
        numberOfSequences,
        sequencesWithFalsePositive,
        totalOrfs,
        falsePositiveOrfs,
    )


def savePositiveControlResults(results: list[PositiveControlResult], outputPath: Path) -> None:
    """Guarda una tabla para comparar cada resultado con el CDS esperado."""
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    with outputPath.open("w", newline="", encoding="utf-8") as outputFile:
        writer = csv.writer(outputFile, delimiter="\t")
        writer.writerow([
            "gen", "accession", "longitud_CDS_esperada_aa", "ORFs_detectados",
            "candidatos_posterior_superior_al_umbral", "rango_CDS_esperado_top5", "acierto",
        ])
        for result in results:
            writer.writerow([
                result.geneName,
                result.accession,
                result.expectedLength,
                result.detectedOrfs,
                result.candidatesAboveThreshold,
                result.expectedOrfRank if result.expectedOrfRank is not None else "no aparece",
                "si" if result.isHit else "no",
            ])


def plotControlResults(
    positiveResults: list[PositiveControlResult],
    negativeResult: NegativeControlResult,
    threshold: float,
    outputPath: Path,
) -> None:
    """Genera un resumen visual de exactitud positiva y falsos positivos."""
    positiveAccuracy = 100 * sum(result.isHit for result in positiveResults) / len(positiveResults)
    falsePositiveRate = 100 * negativeResult.falsePositiveSequenceRate

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.35]})
    figure.patch.set_facecolor("#f7f7f5")
    codingColor = "#2a6f97"
    randomColor = "#d1495b"

    labels = ["Acierto\ncontroles positivos", "Falsos positivos\nADN aleatorio"]
    values = [positiveAccuracy, falsePositiveRate]
    bars = axes[0].bar(labels, values, color=[codingColor, randomColor], width=0.62)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Porcentaje de secuencias (%)")
    axes[0].set_title("Rendimiento global", fontweight="bold")
    annotations = [
        f"{sum(result.isHit for result in positiveResults)}/{len(positiveResults)}\n{positiveAccuracy:.1f}%",
        f"{negativeResult.sequencesWithFalsePositive}/{negativeResult.sequencesTested}\n{falsePositiveRate:.1f}%",
    ]
    axes[0].bar_label(bars, labels=annotations, padding=5, fontsize=11, fontweight="bold")

    ranks = [result.expectedOrfRank if result.expectedOrfRank is not None else 6 for result in positiveResults]
    colors = [codingColor if result.isHit else "#b0b7bc" for result in positiveResults]
    axes[1].barh([result.geneName for result in positiveResults], ranks, color=colors)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 6.4)
    axes[1].set_xticks(range(1, 7), ["#1", "#2", "#3", "#4", "#5", "No esta"])
    axes[1].set_xlabel("Rango del CDS esperado entre candidatos con P(C) > " f"{threshold:g}")
    axes[1].set_title("Control positivo: recuperacion del ORF real", fontweight="bold")
    for index, result in enumerate(positiveResults):
        label = f"#{result.expectedOrfRank}" if result.isHit else "No recuperado"
        axes[1].text(ranks[index] + 0.08, index, label, va="center", fontsize=10)

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)

    figure.suptitle("Controles del detector y clasificador de ORFs", fontsize=17,
                    fontweight="bold", color="#243447")
    figure.tight_layout()
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(outputPath, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def plotSplicingLimitation(
    transcriptResults: list[PositiveControlResult],
    genomicResults: list[PositiveControlResult],
    threshold: float,
    topCount: int,
    outputPath: Path,
) -> None:
    """Visualiza la perdida de recuperacion al usar ADN genomico no empalmado."""
    transcriptAccuracy = 100 * sum(result.isHit for result in transcriptResults) / len(transcriptResults)
    genomicAccuracy = 100 * sum(result.isHit for result in genomicResults) / len(genomicResults)
    genomicByGene = {result.geneName: result for result in genomicResults}

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.35]})
    figure.patch.set_facecolor("#f7f7f5")
    transcriptColor = "#2a6f97"
    genomicColor = "#d1495b"

    labels = ["mRNA\nCDS empalmado", "ADN genomico\nsin splicing"]
    values = [transcriptAccuracy, genomicAccuracy]
    bars = axes[0].bar(labels, values, color=[transcriptColor, genomicColor], width=0.62)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel(f"CDS esperado recuperado en top {topCount} (%)")
    axes[0].set_title("Efecto global del splicing", fontweight="bold")
    axes[0].bar_label(
        bars,
        labels=[
            f"{sum(result.isHit for result in transcriptResults)}/{len(transcriptResults)}\n{transcriptAccuracy:.1f}%",
            f"{sum(result.isHit for result in genomicResults)}/{len(genomicResults)}\n{genomicAccuracy:.1f}%",
        ],
        padding=5,
        fontsize=11,
        fontweight="bold",
    )

    positions = np.arange(len(transcriptResults))
    transcriptHits = [int(result.isHit) for result in transcriptResults]
    genomicHits = [int(genomicByGene[result.geneName].isHit) for result in transcriptResults]
    axes[1].bar(positions - 0.2, transcriptHits, width=0.4, color=transcriptColor,
                label="mRNA (CDS sin intrones)")
    axes[1].bar(positions + 0.2, genomicHits, width=0.4, color=genomicColor,
                label="ADN genomico")
    axes[1].set_xticks(positions, [result.geneName for result in transcriptResults])
    axes[1].set_yticks([0, 1], ["No", "Si"])
    axes[1].set_ylim(0, 1.2)
    axes[1].set_ylabel(f"CDS esperado dentro del top {topCount}")
    axes[1].set_xlabel("Gen humano conocido")
    axes[1].set_title(f"Mismo detector y umbral P(C) > {threshold:g}", fontweight="bold")
    axes[1].legend(frameon=False, loc="upper right")

    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)

    figure.suptitle("Limitacion del detector de ORFs: intrones sin splicing", fontsize=17,
                    fontweight="bold", color="#243447")
    figure.tight_layout()
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(outputPath, dpi=220, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True, help="Email requerido por NCBI Entrez")
    parser.add_argument("--umbral", type=float, default=0.99)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--proteinas", type=int, default=1000)
    parser.add_argument("--muestras-entrenamiento", type=int, default=10)
    parser.add_argument("--longitud-entrenamiento", type=int, default=100_000)
    parser.add_argument("--muestras-negativas", type=int, default=30)
    parser.add_argument("--longitud-negativa", type=int, default=20_000)
    parser.add_argument("--semilla", type=int, default=DEFAULT_SEED)
    parser.add_argument("--salida", type=Path,
                        default=Path(__file__).with_name("figuras") / "controles_orf.png")
    parser.add_argument("--tabla", type=Path,
                        default=Path(__file__).with_name("resultados") / "resultados_control_positivo.tsv")
    parser.add_argument("--salida-splicing", type=Path,
                        default=Path(__file__).with_name("figuras") / "limitacion_splicing.png")
    parser.add_argument("--tabla-genomica", type=Path,
                        default=Path(__file__).with_name("resultados") / "resultados_control_genomico.tsv")
    arguments = parser.parse_args()
    if not 0 < arguments.umbral < 1:
        raise ValueError("El umbral debe estar estrictamente entre 0 y 1.")
    if arguments.top <= 0:
        raise ValueError("La cantidad maxima de candidatos debe ser positiva.")

    realProteins = downloadHumanProteins(
        arguments.email, arguments.proteinas, seed=arguments.semilla
    )
    trainingRandomOrfs = generateRandomOrfs(
        arguments.muestras_entrenamiento,
        arguments.longitud_entrenamiento,
        arguments.semilla,
    )
    classifier = trainOrfClassifier(realProteins, trainingRandomOrfs)

    positiveResults = runPositiveControls(
        arguments.email, classifier, arguments.umbral, arguments.top
    )
    genomicResults = runGenomicControls(
        arguments.email, classifier, arguments.umbral, arguments.top
    )
    negativeResult = runNegativeControl(
        classifier,
        arguments.muestras_negativas,
        arguments.longitud_negativa,
        arguments.umbral,
        arguments.semilla + 1,
    )
    savePositiveControlResults(positiveResults, arguments.tabla)
    savePositiveControlResults(genomicResults, arguments.tabla_genomica)
    plotControlResults(positiveResults, negativeResult, arguments.umbral, arguments.salida)
    plotSplicingLimitation(
        positiveResults, genomicResults, arguments.umbral, arguments.top,
        arguments.salida_splicing
    )

    hits = sum(result.isHit for result in positiveResults)
    genomicHits = sum(result.isHit for result in genomicResults)
    print(f"Control positivo: {hits}/{len(positiveResults)} CDS recuperados en el top {arguments.top}.")
    print(
        "Control en ADN genomico sin splicing: "
        f"{genomicHits}/{len(genomicResults)} CDS recuperados en el top {arguments.top}."
    )
    print(
        "Control negativo: "
        f"{negativeResult.sequencesWithFalsePositive}/{negativeResult.sequencesTested} "
        "secuencias aleatorias tuvieron al menos un falso positivo."
    )
    print(f"Tabla guardada en: {arguments.tabla.resolve()}")
    print(f"Tabla genomica guardada en: {arguments.tabla_genomica.resolve()}")
    print(f"Grafico guardado en: {arguments.salida.resolve()}")
    print(f"Grafico de splicing guardado en: {arguments.salida_splicing.resolve()}")


if __name__ == "__main__":
    main()
