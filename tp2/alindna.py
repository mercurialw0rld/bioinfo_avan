from Bio import Entrez, SeqIO, ExPASy, SwissProt
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator


def descargaProteina(id_proteina):
    """Descarga la secuencia de una proteína desde UniProt."""
    try:
        handle = ExPASy.get_sprot_raw(id_proteina)
        record = SwissProt.read(handle)
        print(f"Descargando {record.entry_name} ({record.accessions[0]})")
        sequence = record.sequence
        return SeqRecord(Seq(sequence), id=id_proteina, description=record.description)
    except Exception as e:
        print(f"Error al descargar la proteína {id_proteina}: {e}")
        return None

def descargaADN(email, geneName, organism):
    """Descarga el intervalo de ADN genómico anotado para un gen.

    Busca el gen en la base Gene, obtiene el cromosoma y las coordenadas de
    su ensamblado de referencia y descarga únicamente ese intervalo. La
    secuencia se devuelve en la orientación 5' -> 3' del gen, no como mRNA.
    """
    Entrez.email = email
    searchQuery = f'"{geneName}"[Gene Name] AND "{organism}"[Organism]'
    with Entrez.esearch(db="gene", term=searchQuery, retmax=20) as searchHandle:
        geneIds = Entrez.read(searchHandle)["IdList"]
    if not geneIds:
        raise ValueError(f"No se encontró el gen {geneName} en {organism}.")

    with Entrez.esummary(db="gene", id=",".join(geneIds)) as summaryHandle:
        summaries = Entrez.read(summaryHandle)["DocumentSummarySet"]["DocumentSummary"]

    matchingSummary = next(
        (
            summary
            for summary in summaries
            if summary["Name"].casefold() == geneName.casefold()
            and summary["Organism"]["ScientificName"].casefold() == organism.casefold()
        ),
        None,
    )
    if matchingSummary is None:
        raise ValueError(f"No se encontró una coincidencia exacta para {geneName} en {organism}.")

    genomicInfo = matchingSummary["GenomicInfo"]
    if not genomicInfo:
        raise ValueError(f"{geneName} no tiene coordenadas genómicas RefSeq anotadas.")

    location = genomicInfo[0]
    chromosomeAccession = location["ChrAccVer"]
    chromosomeStart = int(location["ChrStart"])
    chromosomeStop = int(location["ChrStop"])

    # Gene informa posiciones 0-based; efetch espera posiciones 1-based.
    sequenceStart = min(chromosomeStart, chromosomeStop) + 1
    sequenceStop = max(chromosomeStart, chromosomeStop) + 1
    strand = 1 if chromosomeStart < chromosomeStop else 2

    with Entrez.efetch(
        db="nuccore",
        id=chromosomeAccession,
        seq_start=sequenceStart,
        seq_stop=sequenceStop,
        strand=strand,
        rettype="gb",
        retmode="text",
    ) as recordHandle:
        genomicRecord = SeqIO.read(recordHandle, "genbank")
    return genomicRecord

def match(seq1, seq2, threshold):
    """Compara dos secuencias y devuelve el número de coincidencias."""
    return sum(1 for a, b in zip(seq1, seq2) if a == b)

def crearMatrizDot(seq1, seq2, window, threshold, filter):
    """Crea una matriz de puntos (dot matrix) para comparar dos secuencias."""
    if filter == 'solapados':
        len_seq1 = len(seq1) - window + 1
        len_seq2 = len(seq2) - window + 1
        matriz = np.zeros((len_seq1, len_seq2), dtype=int)
        for i in range(len_seq1):
            seq1_window = seq1[i:i + window]
            for j in range(len_seq2):
                seq2_window = seq2[j:j + window]
                if match(seq1_window, seq2_window, threshold) >= threshold:
                    matriz[i, j] = 1
        return matriz
    elif filter == 'adyreduccion':
        len_seq1 = int(len(seq1) // window)
        len_seq2 = int(len(seq2) // window)
        matriz = np.zeros((len_seq1, len_seq2), dtype=int)
        for i in range(len_seq1):
            seq1_window = seq1[i * window:(i + 1) * window]
            for j in range(len_seq2):
                seq2_window = seq2[j * window:(j + 1) * window]
                if match(seq1_window, seq2_window, threshold) >= threshold:
                    matriz[i, j] = 1
        return matriz
    elif filter == 'adynoreduccion':
        len_seq1 = len(seq1)
        len_seq2 = len(seq2)
        # hacemos la matriz completa de coincidencias
        matrizCompleta = np.zeros((len_seq1, len_seq2), dtype=int)
        for i in range(0, len_seq1):
            for j in range(0, len_seq2):
                if seq1[i] == seq2[j]:
                    matrizCompleta[i, j] = 1

        # ahora aplicamos la reducción adyacente
        for i in range(0, len_seq1, window):
            seq1_window = seq1[i:i + window]
            for j in range(0, len_seq2, window):
                seq2_window = seq2[j:j + window]
                if match(seq1_window, seq2_window, threshold) < threshold:
                    for n in range(i, min(i + window, len_seq1)):
                        for m in range(j, min(j + window, len_seq2)):
                            matrizCompleta[n, m] = 0
        return matrizCompleta
    else:
        raise ValueError("El parámetro 'filter' debe ser 'solapados', 'adyreduccion' o 'adynoreduccion'.")

def recorrerMatrizDot(matriz, filterName="sin especificar"):
    """Recorre el dot plot mediante alineamiento global."""

    gapPenalty = -1
    score = 1

    dp = np.zeros(
        (matriz.shape[0] + 1, matriz.shape[1] + 1),
        dtype=int
    )

    # Inicialización global
    for i in range(1, dp.shape[0]):
        dp[i, 0] = i * gapPenalty

    for j in range(1, dp.shape[1]):
        dp[0, j] = j * gapPenalty

    # Llenado
    for i in range(1, dp.shape[0]):
        for j in range(1, dp.shape[1]):

            diagonal = dp[i - 1, j - 1] + matriz[i - 1, j - 1]
            arriba = dp[i - 1, j] + gapPenalty
            izquierda = dp[i, j - 1] + gapPenalty

            dp[i, j] = max(diagonal, arriba, izquierda)

    # Traceback desde esquina inferior derecha
    i = matriz.shape[0]
    j = matriz.shape[1]

    ruta = []

    while i > 0 or j > 0:

        if (
            i > 0 and j > 0
            and dp[i, j]
            == dp[i - 1, j - 1] + matriz[i - 1, j - 1]
        ):
            ruta.append((i - 1, j - 1))
            score += 1
            i -= 1
            j -= 1

        elif (
            i > 0
            and dp[i, j] == dp[i - 1, j] + gapPenalty
        ):
            score -= 1
            ruta.append((i - 1, j))
            i -= 1

        else:
            score -= 1
            ruta.append((i, j - 1))
            j -= 1

    ruta.reverse()
    print(f"Score de la mejor ruta ({filterName}): {score}")

    return ruta


def configurarEjesDot(ax, matriz, seq1, seq2, window=1, reduced=False):
    """Configura ejes de un dot plot en coordenadas de nucleótidos."""
    cellSize = window if reduced else 1
    xLimit = matriz.shape[1] * cellSize - cellSize / 2
    yLimit = matriz.shape[0] * cellSize - cellSize / 2
    ax.imshow(
        matriz,
        cmap=ListedColormap(["#f7f7f7", "#1f5a7a"]),
        vmin=0,
        vmax=1,
        interpolation="none",
        extent=(-cellSize / 2, xLimit, yLimit, -cellSize / 2),
        aspect="equal",
    )
    ax.set_xlabel("Posición en secuencia 2 (nt)")
    ax.set_ylabel("Posición en secuencia 1 (nt)")
    if reduced:
        # Se muestran como máximo seis marcas para evitar que se superpongan.
        xIndexes = np.linspace(0, matriz.shape[1] - 1, min(6, matriz.shape[1]), dtype=int)
        yIndexes = np.linspace(0, matriz.shape[0] - 1, min(6, matriz.shape[0]), dtype=int)
        ax.set_xticks(xIndexes * window)
        ax.set_yticks(yIndexes * window)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.ticklabel_format(style="plain", axis="both")
    ax.tick_params(axis="both", labelsize=8)


def guardarFigura(fig, fileName, outputDirectory):
    """Guarda una figura PNG en el directorio de resultados del trabajo práctico."""
    figuresDirectory = Path(__file__).parent / "figuras" / outputDirectory
    figuresDirectory.mkdir(exist_ok=True)
    fig.savefig(figuresDirectory / fileName, dpi=300)


def graficarDot(matriz, seq1, seq2, outputDirectory):
    """Grafica una matriz de puntos con ejes en posiciones de nucleótidos."""
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    configurarEjesDot(ax, matriz, seq1, seq2)
    ax.set_title("Dot plot de coincidencias", fontweight="bold")
    ax.text(
        0.5,
        -0.16,
        "Cada punto representa una coincidencia según el filtro elegido.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    guardarFigura(fig, "dot_plot.png", outputDirectory)
    plt.show()

def graficarDotConRuta(matriz, ruta, seq1, seq2, outputDirectory):
    fig, ax = plt.subplots(figsize=(8, 8))

    configurarEjesDot(ax, matriz, seq1, seq2)

    # Ruta
    x = [j for i, j in ruta]
    y = [i for i, j in ruta]

    ax.plot(x, y, linewidth=1.5, color="orange", label="Mejor ruta")

    ax.set_title("Dot-plot con mejor camino")
    plt.tight_layout()
    guardarFigura(fig, "dot_plot_con_ruta.png", outputDirectory)
    plt.show()

def graficarTresDots(matrices, titulos, seq1, seq2, outputDirectory, window=1):
    """Grafica tres dot plots comparables con las mismas escalas de ejes."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)

    for index, (ax, matriz, titulo) in enumerate(zip(axes, matrices, titulos)):
        configurarEjesDot(
            ax,
            matriz,
            seq1,
            seq2,
            window=window,
            reduced=titulo == "Adyacente con reducción",
        )
        ax.set_title(titulo, fontweight="bold")
        if index > 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

    fig.suptitle("Comparación de filtros para el dot plot", fontsize=15, fontweight="bold")
    guardarFigura(fig, "comparacion_filtros.png", outputDirectory)
    plt.show()

def graficarRutaTresDots(matrices, rutas, titulos, seq1, seq2, outputDirectory, window=1):
    """Grafica tres dot plots comparables con las mismas escalas de ejes y sus rutas."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)

    for index, (ax, matriz, ruta, titulo) in enumerate(zip(axes, matrices, rutas, titulos)):
        isReduced = titulo == "Adyacente con reducción"
        configurarEjesDot(ax, matriz, seq1, seq2, window=window, reduced=isReduced)
        ax.set_title(titulo, fontweight="bold")
        if index > 0:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        coordinateScale = window if isReduced else 1
        x = [j * coordinateScale for i, j in ruta]
        y = [i * coordinateScale for i, j in ruta]
        ax.plot(x, y, linewidth=1.5, color="orange", label="Mejor ruta")

    fig.suptitle("Comparación de filtros para el dot plot con rutas", fontsize=15, fontweight="bold")
    guardarFigura(fig, "comparacion_filtros_con_rutas.png", outputDirectory)
    plt.show()

def analizarPar(seq1, seq2, outputDirectory):
        dotMatrix = crearMatrizDot(seq1, seq2, window=10, threshold=7, filter='solapados')

        graficarDot(dotMatrix, seq1, seq2, outputDirectory)
        dotMatrixAdy = crearMatrizDot(seq1, seq2, window=10, threshold=7, filter='adynoreduccion')
        dotMatrixAdyR = crearMatrizDot(seq1, seq2, window=10, threshold=7, filter='adyreduccion')
        graficarTresDots(
            [dotMatrix, dotMatrixAdy, dotMatrixAdyR],
            ['Solapados', 'Adyacente sin reducción', 'Adyacente con reducción'],
            seq1,
            seq2,
            outputDirectory,
            window=10,
        )
        ruta = recorrerMatrizDot(dotMatrixAdy, "Adyacente sin reduccion")
        graficarDotConRuta(dotMatrixAdy, ruta, seq1, seq2, outputDirectory)
        graficarRutaTresDots(
            [dotMatrix, dotMatrixAdy, dotMatrixAdyR],
            [
                recorrerMatrizDot(dotMatrix, "Solapados"),
                ruta,
                recorrerMatrizDot(dotMatrixAdyR, "Adyacente con reduccion"),
            ],
            ['Solapados', 'Adyacente sin reducción', 'Adyacente con reducción'],
            seq1,
            seq2,
            outputDirectory,
            window=10,
        )
        return None

if __name__ == "__main__":
    email = "facundokisielus@gmail.com"
    genes = [{"name": "HBB", "organism": "Homo sapiens"}, {"name": "HBB", "organism": "Pan troglodytes"}, {"name": "HBB", "organism": "Lepus europaeus"}]
    print("Descargando genes...")
    seqs = []

    for gene in genes:
        try:
            record = descargaADN(email, gene["name"], gene["organism"])
            print(f"Descargado {record.id}: {record.description}")
            seqs.append(record.seq)
        except Exception as e:
            print(f"Error al descargar el gen {gene['name']}: {e}")

    try:
        # HBB Humana vs Chimpance
        analizarPar(seqs[0], seqs[1], "HBB_Humana_vs_Chimpance")
        # HBB humana vs Raton
        analizarPar(seqs[0], seqs[2], "HBB_Humana_vs_Liebre")

    except ValueError as e:
        print(f"Error al crear la matriz de puntos: {e}")
