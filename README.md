# Generatore Verbale - Luglio

## Descrizione

Questa applicazione consente di:
- Caricare file audio lunghi (wav, mp3, m4a)
- Trascrivere automaticamente il contenuto (openai-whisper)
- Separare i parlanti (diarizzazione, whisperx)
- Generare un riassunto automatico (transformers)
- Visualizzare tutto tramite una semplice interfaccia web (Streamlit)

## Requisiti

- Windows 10/11
- Python 3.9 o superiore (consigliato Python 3.10)
- ffmpeg installato e nel PATH di sistema

## Installazione e Avvio

1. **Scarica o clona questa cartella.**
2. **Apri il prompt dei comandi nella cartella `Generatore-Verbale-luglio`.**
3. **Esegui il file `setup.bat`**:
   ```
   setup.bat
   ```
   Il batch:
   - Crea una virtualenv se non esiste
   - Installa tutte le dipendenze
   - Avvia l'applicazione Streamlit

4. **Segui le istruzioni a schermo per caricare il file audio e ottenere trascrizione, diarizzazione e riassunto.**

## Output

- Le trascrizioni vengono salvate in `output/trascrizioni/`
- I riassunti in `output/riassunti/`

## Note

- Per file molto lunghi la trascrizione può richiedere diversi minuti.
- Se la diarizzazione non è precisa, prova a usare file audio di qualità migliore.
- Per problemi con ffmpeg, assicurati che sia installato e accessibile dal prompt dei comandi (`ffmpeg -version`).

## Dipendenze principali

- openai-whisper
- whisperx
- transformers
- streamlit
- torchaudio
- pydub
