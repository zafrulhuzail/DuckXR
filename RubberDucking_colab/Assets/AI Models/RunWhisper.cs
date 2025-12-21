using System.Collections.Generic;
using UnityEngine;
using Unity.InferenceEngine;
using System.Text;
using Unity.Collections;
using Newtonsoft.Json;
using TMPro;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;
public class RunWhisper : MonoBehaviour
{
    Worker decoder1, decoder2, encoder, spectrogram;
    Worker argmax;

    [Header("Input mode")]
    public bool useMicrophone = false;          // toggle in Inspector
    [Tooltip("Seconds to record when using the microphone")]
    public int micRecordSeconds = 5;

    const int sampleRate = 16000;              // Whisper expects 16 kHz

    [Header("Recording Control")]
    public bool startRecording = false; // toggle this to trigger recording
    private bool isRecording = false;   // internal safety to prevent double calls

    [Header("Recording Control")]
    public KeyCode recordKey = KeyCode.R;   // press R to start recording

    [Header("Transcription Targets")]
    public TMP_Text[] transcriptionSlots;   // assign in Inspector
    private TMP_Text currentSlot;

public int activeSlot = 0;
    private AudioClip micClip;
    public AudioClip audioClip;
    public TMP_Text whisperText;

    // This is how many tokens you want. It can be adjusted.
    const int maxTokens = 100;

    // Special tokens see added tokens file for details
    const int END_OF_TEXT = 50257;
    const int START_OF_TRANSCRIPT = 50258;
    const int ENGLISH = 50259;
    const int GERMAN = 50261;
    const int FRENCH = 50265;
    const int TRANSCRIBE = 50359; //for speech-to-text in specified language
    const int TRANSLATE = 50358;  //for speech-to-text then translate to English
    const int NO_TIME_STAMPS = 50363;
    const int START_TIME = 50364;

    int numSamples;
    string[] tokens;

    int tokenCount = 0;
    NativeArray<int> outputTokens;

    // Used for special character decoding
    int[] whiteSpaceCharacters = new int[256];

    Tensor<float> encodedAudio;

    bool transcribe = false;
    string outputString = "";

    // Maximum size of audioClip (30s at 16kHz)
    const int maxSamples = 30 * 16000;

    public ModelAsset audioDecoder1, audioDecoder2;
    public ModelAsset audioEncoder;
    public ModelAsset logMelSpectro;

    private StringBuilder currentSentence = new StringBuilder();
    private List<string> bulletPoints = new List<string>();

    public async void Start()
    {
        Debug.Log($"Clip: {audioClip.name}, length: {audioClip.length:F2}s, freq: {audioClip.frequency} Hz, channels: {audioClip.channels}");
        Debug.Log("Unity microphones: " + string.Join(", ", Microphone.devices));
        currentSlot = CurrentTextSlot;
        whisperText = currentSlot;
        SetupWhiteSpaceShifts();
        GetTokens();

        Debug.Log("Ready. Awaiting recording trigger...");

        decoder1 = new Worker(ModelLoader.Load(audioDecoder1), BackendType.GPUCompute);
        decoder2 = new Worker(ModelLoader.Load(audioDecoder2), BackendType.GPUCompute);

        FunctionalGraph graph = new FunctionalGraph();
        var input = graph.AddInput(DataType.Float, new DynamicTensorShape(1, 1, 51865));
        var amax = Functional.ArgMax(input, -1, false);
        var selectTokenModel = graph.Compile(amax);
        argmax = new Worker(selectTokenModel, BackendType.GPUCompute);

        encoder = new Worker(ModelLoader.Load(audioEncoder), BackendType.GPUCompute);
        spectrogram = new Worker(ModelLoader.Load(logMelSpectro), BackendType.GPUCompute);

        outputTokens = new NativeArray<int>(maxTokens, Allocator.Persistent);

        outputTokens[0] = START_OF_TRANSCRIPT;
        outputTokens[1] = ENGLISH;// GERMAN;//FRENCH;//
        outputTokens[2] = TRANSCRIBE; //TRANSLATE;//
        //outputTokens[3] = NO_TIME_STAMPS;// START_TIME;//
        tokenCount = 3;

        Debug.Log("Whisper ready. Press " + recordKey + " to record from mic.");
    }

    private void Update()
    {
        whisperText = currentSlot;
        Debug.Log("Active Slot: " + activeSlot);
        Debug.Log("Entering Update function");
        // Debug.Log("Value of " + audioInput.GetKeyDown(recordKey));
        Debug.Log("Value of " + isRecording);
        // if (!useMicrophone)
        //     return; // only use keyboard when mic mode is on

        if (Input.GetKeyDown(recordKey) && !isRecording)
        {
            StartMicTranscription();
            // isRecording = false;
        }
    }
    TMP_Text CurrentTextSlot
    {
        get
        {
            if (activeSlot < 0 || activeSlot >= transcriptionSlots.Length)
                return null;
            return transcriptionSlots[activeSlot];
        }
    }
    async void StartMicTranscription()
    {
        isRecording = true;

        if (whisperText != null)
            whisperText.text = "Listening...";

        if (useMicrophone)
        {
            await RecordFromMicrophone();          // record into micClip + LoadAudioFromClip
        }
        else
        {
            if (audioClip == null)
            {
                Debug.LogError("No AudioClip assigned and useMicrophone is false.");
                return;
            }

            LoadAudio(audioClip);          // old behaviour
        }
        EncodeAudio();
        transcribe = true;

        tokensTensor = new Tensor<int>(new TensorShape(1, maxTokens));
        ComputeTensorData.Pin(tokensTensor);
        tokensTensor.Reshape(new TensorShape(1, tokenCount));
        tokensTensor.dataOnBackend.Upload<int>(outputTokens, tokenCount);

        lastToken = new NativeArray<int>(1, Allocator.Persistent); lastToken[0] = NO_TIME_STAMPS;
        lastTokenTensor = new Tensor<int>(new TensorShape(1, 1), new[] { NO_TIME_STAMPS });

        while (true)
        {
            if (!transcribe || tokenCount >= (outputTokens.Length - 1))
                break;
            m_Awaitable = InferenceStep();
            await m_Awaitable;
        }

        isRecording = false;
    }


    Awaitable m_Awaitable;

    NativeArray<int> lastToken;
    Tensor<int> lastTokenTensor;
    Tensor<int> tokensTensor;
    Tensor<float> audioInput;

    // void LoadAudio()
    // {
    //     numSamples = audioClip.samples;
    //     var data = new float[maxSamples];
    //     numSamples = maxSamples;
    //     audioClip.GetData(data, 0);
    //     audioInput = new Tensor<float>(new TensorShape(1, numSamples), data);
    // }

    void LoadAudio(AudioClip clip)
    {
        if (clip == null)
        {
            Debug.LogError("LoadAudioFromClip: clip is null");
            return;
        }

        // Always use 30s window, pad with zeros if clip is shorter
        var data = new float[maxSamples];
        numSamples = maxSamples;

        clip.GetData(data, 0); // Unity will just copy what exists

        audioInput = new Tensor<float>(new TensorShape(1, numSamples), data);

        Debug.Log($"Loaded audio from {clip.name}, length={clip.length:F2}s, " +
                $"freq={clip.frequency} Hz, channels={clip.channels}");
    }

    async Task RecordFromMicrophone()
    {
        if (Microphone.devices.Length == 0)
        {
            Debug.LogError("No microphone devices found!");
            return;
        }

        string micName = Microphone.devices[0];

        Debug.Log($"Recording from mic '{micName}' for {micRecordSeconds} seconds @ {sampleRate} Hz");

        // Optional: UI feedback
        if (whisperText != null)
            whisperText.text = "Listening...";

        micClip = Microphone.Start(micName, false, micRecordSeconds, sampleRate);

        // Wait until recording starts
        while (Microphone.GetPosition(micName) <= 0) { }

        // Wait for the desired duration
        await Task.Delay(micRecordSeconds * 1000);

        Microphone.End(micName);

        LoadAudio(micClip);
    }



    void EncodeAudio()
    {
        spectrogram.Schedule(audioInput);
        var logmel = spectrogram.PeekOutput() as Tensor<float>;
        encoder.Schedule(logmel);
        encodedAudio = encoder.PeekOutput() as Tensor<float>;
    }
    async Awaitable InferenceStep()
    {
        decoder1.SetInput("input_ids", tokensTensor);
        decoder1.SetInput("encoder_hidden_states", encodedAudio);
        decoder1.Schedule();

        var past_key_values_0_decoder_key = decoder1.PeekOutput("present.0.decoder.key") as Tensor<float>;
        var past_key_values_0_decoder_value = decoder1.PeekOutput("present.0.decoder.value") as Tensor<float>;
        var past_key_values_1_decoder_key = decoder1.PeekOutput("present.1.decoder.key") as Tensor<float>;
        var past_key_values_1_decoder_value = decoder1.PeekOutput("present.1.decoder.value") as Tensor<float>;
        var past_key_values_2_decoder_key = decoder1.PeekOutput("present.2.decoder.key") as Tensor<float>;
        var past_key_values_2_decoder_value = decoder1.PeekOutput("present.2.decoder.value") as Tensor<float>;
        var past_key_values_3_decoder_key = decoder1.PeekOutput("present.3.decoder.key") as Tensor<float>;
        var past_key_values_3_decoder_value = decoder1.PeekOutput("present.3.decoder.value") as Tensor<float>;

        var past_key_values_0_encoder_key = decoder1.PeekOutput("present.0.encoder.key") as Tensor<float>;
        var past_key_values_0_encoder_value = decoder1.PeekOutput("present.0.encoder.value") as Tensor<float>;
        var past_key_values_1_encoder_key = decoder1.PeekOutput("present.1.encoder.key") as Tensor<float>;
        var past_key_values_1_encoder_value = decoder1.PeekOutput("present.1.encoder.value") as Tensor<float>;
        var past_key_values_2_encoder_key = decoder1.PeekOutput("present.2.encoder.key") as Tensor<float>;
        var past_key_values_2_encoder_value = decoder1.PeekOutput("present.2.encoder.value") as Tensor<float>;
        var past_key_values_3_encoder_key = decoder1.PeekOutput("present.3.encoder.key") as Tensor<float>;
        var past_key_values_3_encoder_value = decoder1.PeekOutput("present.3.encoder.value") as Tensor<float>;

        decoder2.SetInput("input_ids", lastTokenTensor);
        decoder2.SetInput("past_key_values.0.decoder.key", past_key_values_0_decoder_key);
        decoder2.SetInput("past_key_values.0.decoder.value", past_key_values_0_decoder_value);
        decoder2.SetInput("past_key_values.1.decoder.key", past_key_values_1_decoder_key);
        decoder2.SetInput("past_key_values.1.decoder.value", past_key_values_1_decoder_value);
        decoder2.SetInput("past_key_values.2.decoder.key", past_key_values_2_decoder_key);
        decoder2.SetInput("past_key_values.2.decoder.value", past_key_values_2_decoder_value);
        decoder2.SetInput("past_key_values.3.decoder.key", past_key_values_3_decoder_key);
        decoder2.SetInput("past_key_values.3.decoder.value", past_key_values_3_decoder_value);

        decoder2.SetInput("past_key_values.0.encoder.key", past_key_values_0_encoder_key);
        decoder2.SetInput("past_key_values.0.encoder.value", past_key_values_0_encoder_value);
        decoder2.SetInput("past_key_values.1.encoder.key", past_key_values_1_encoder_key);
        decoder2.SetInput("past_key_values.1.encoder.value", past_key_values_1_encoder_value);
        decoder2.SetInput("past_key_values.2.encoder.key", past_key_values_2_encoder_key);
        decoder2.SetInput("past_key_values.2.encoder.value", past_key_values_2_encoder_value);
        decoder2.SetInput("past_key_values.3.encoder.key", past_key_values_3_encoder_key);
        decoder2.SetInput("past_key_values.3.encoder.value", past_key_values_3_encoder_value);

        decoder2.Schedule();

        var logits = decoder2.PeekOutput("logits") as Tensor<float>;
        argmax.Schedule(logits);
        using var t_Token = await argmax.PeekOutput().ReadbackAndCloneAsync() as Tensor<int>;
        int index = t_Token[0];

        outputTokens[tokenCount] = lastToken[0];
        lastToken[0] = index;
        tokenCount++;
        tokensTensor.Reshape(new TensorShape(1, tokenCount));
        tokensTensor.dataOnBackend.Upload<int>(outputTokens, tokenCount);
        lastTokenTensor.dataOnBackend.Upload<int>(lastToken, 1);

        if (index == END_OF_TEXT)
        {
            transcribe = false;

            // Flush any unfinished sentence as a final bullet
            if (currentSentence.Length > 0)
            {
                string chunk = CleanChunk(currentSentence.ToString());
                if (!string.IsNullOrWhiteSpace(chunk))
                    bulletPoints.Add("• " + chunk);

                currentSentence.Clear();
            }

            // Show all bullets
            if (whisperText != null)
                whisperText.text = string.Join("\n", bulletPoints);

            Debug.Log("Whisper bullets:\n" + whisperText.text);
        }
        else if (index < tokens.Length)
        {
            string tokenText = GetUnicodeText(tokens[index]);

            // keep full raw text if you still need it
            outputString += tokenText;

            // build current sentence
            currentSentence.Append(tokenText);

            // did this token end a sentence?
            if (Regex.IsMatch(tokenText, @"[.?!]"))
            {
                string chunk = CleanChunk(currentSentence.ToString());
                if (!string.IsNullOrWhiteSpace(chunk))
                    bulletPoints.Add("• " + chunk);

                currentSentence.Clear();
            }

            // update UI with all bullets so far
            if (whisperText != null)
                whisperText.text = string.Join("\n", bulletPoints);
        }

        // Debug.Log(outputString);
        Debug.Log("Whisper: " + outputString);

    }

    // Tokenizer
    public TextAsset vocabAsset;
    void GetTokens()
    {
        var vocab = JsonConvert.DeserializeObject<Dictionary<string, int>>(vocabAsset.text);
        tokens = new string[vocab.Count];
        foreach (var item in vocab)
        {
            tokens[item.Value] = item.Key;
        }
    }

    string GetUnicodeText(string text)
    {
        var bytes = Encoding.GetEncoding("ISO-8859-1").GetBytes(ShiftCharacterDown(text));
        return Encoding.UTF8.GetString(bytes);
    }

    string ShiftCharacterDown(string text)
    {
        string outText = "";
        foreach (char letter in text)
        {
            outText += ((int)letter <= 256) ? letter : (char)whiteSpaceCharacters[(int)(letter - 256)];
        }
        return outText;
    }

    void SetupWhiteSpaceShifts()
    {
        for (int i = 0, n = 0; i < 256; i++)
        {
            if (IsWhiteSpace((char)i)) whiteSpaceCharacters[n++] = i;
        }
    }

    bool IsWhiteSpace(char c)
    {
        return !(('!' <= c && c <= '~') || ('�' <= c && c <= '�') || ('�' <= c && c <= '�'));
    }

    private void OnDestroy()
    {
        decoder1.Dispose();
        decoder2.Dispose();
        encoder.Dispose();
        spectrogram.Dispose();
        argmax.Dispose();
        audioInput.Dispose();
        lastTokenTensor.Dispose();
        tokensTensor.Dispose();
    }
    string CleanChunk(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return string.Empty;

        // collapse repeated spaces
        text = Regex.Replace(text, @"\s+", " ");

        // fix "word ," → "word,"
        text = Regex.Replace(text, @"\s+([.,!?;:])", "$1");

        text = text.Trim();

        // Capitalize first letter
        if (text.Length > 0)
            text = char.ToUpper(text[0]) + text.Substring(1);

        return text;
    }

    public void TriggerRecordingFromInteraction()
    {
        if (!isRecording)
        {
            Debug.Log("TriggerRecordingFromInteraction called");
            StartMicTranscription();
        }
        else
        {
            Debug.Log("Whisper is already recording, ignoring trigger.");
        }
    }

}
