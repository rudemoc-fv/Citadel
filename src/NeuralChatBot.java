import java.util.*;
import java.io.*;
import java.nio.file.*;

public class NeuralChatBot {
    private int vocabSize;
    private final int hiddenSize;
    private final double learningRate;
    private double[][] Wf, Wi, Wc, Wo, Wy;
    private final Map<String, Integer> vocab = new HashMap<>();
    private final List<String> words = new ArrayList<>();
    private final Random random = new Random();
    private final Scanner scanner = new Scanner(System.in);

    public NeuralChatBot(int hiddenSize, double learningRate) {
        this.hiddenSize = hiddenSize;
        this.learningRate = learningRate;
    }

    private void initializeWeights() {
        int totalSize = vocabSize + hiddenSize;
        Wf = createMatrix(hiddenSize, totalSize);
        Wi = createMatrix(hiddenSize, totalSize);
        Wc = createMatrix(hiddenSize, totalSize);
        Wo = createMatrix(hiddenSize, totalSize);
        Wy = createMatrix(vocabSize, hiddenSize);
    }

    private double[][] createMatrix(int rows, int cols) {
        double[][] m = new double[rows][cols];
        double scale = Math.sqrt(2.0 / (rows + cols));
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                m[i][j] = random.nextGaussian() * scale;
            }
        }
        return m;
    }

    static class LSTMState {  // Убрано private
        double[] h;
        double[] c;
        double[] output;
        double[] f;
        double[] i;
        double[] c_hat;
        double[] o;
        double[] xh;

        // Явный конструктор
        public LSTMState(int hiddenSize, int vocabSize) {
            h = new double[hiddenSize];
            c = new double[hiddenSize];
            output = new double[vocabSize];
            f = new double[hiddenSize];
            i = new double[hiddenSize];
            c_hat = new double[hiddenSize];
            o = new double[hiddenSize];
            xh = new double[hiddenSize + vocabSize];
        }
    }


    private LSTMState lstmStep(double[] x, LSTMState prevState) {
    	LSTMState state = new LSTMState(hiddenSize, vocabSize);
        state.xh = concatenate(x, prevState.h);

        // 1. Вычисление всех гейтов
        state.f = sigmoid(mvMult(Wf, state.xh)); // Forget gate
        state.i = sigmoid(mvMult(Wi, state.xh)); // Input gate
        state.c_hat = tanh(mvMult(Wc, state.xh)); // Cell state proposal
        state.o = sigmoid(mvMult(Wo, state.xh)); // Output gate

        // 2. Обновление состояния ячейки
        state.c = add(
            multiply(state.f, prevState.c), 
            multiply(state.i, state.c_hat)
        );

        // 3. Обновление скрытого состояния
        state.h = multiply(state.o, tanh(state.c));

        // 4. Выход сети
        state.output = softmax(mvMult(Wy, state.h));

        return state;
    }

    private void backward(LSTMState state, LSTMState prevState, double[] target) {
        double[] dy = subtract(state.output, target);

        // 1. Градиенты для Wy
        for (int row = 0; row < Wy.length; row++) {
            for (int col = 0; col < Wy[row].length; col++) {
                Wy[row][col] -= learningRate * dy[row] * state.h[col];
            }
        }

        // 2. Градиенты через выходной слой
        double[] dh = mvMultTranspose(Wy, dy);

        // 3. Градиенты для output gate
        double[] do_grad = multiply(dh, tanh(state.c));
        do_grad = multiply(do_grad, sigmoidGradient(state.o));
        double[][] dWo = outerProduct(do_grad, state.xh);

        // 4. Градиенты для cell state
        double[] dc = multiply(dh, multiply(state.o, tanhGradient(state.c)));
        
        // 5. Градиенты для forget gate
        double[] df_grad = multiply(dc, prevState.c);
        df_grad = multiply(df_grad, sigmoidGradient(state.f));
        double[][] dWf = outerProduct(df_grad, state.xh);

        // 6. Градиенты для input gate
        double[] di_grad = multiply(dc, state.c_hat);
        di_grad = multiply(di_grad, sigmoidGradient(state.i));
        double[][] dWi = outerProduct(di_grad, state.xh);

        // 7. Градиенты для cell state proposal
        double[] dc_hat_grad = multiply(dc, state.i);
        dc_hat_grad = multiply(dc_hat_grad, tanhGradient(state.c_hat));
        double[][] dWc = outerProduct(dc_hat_grad, state.xh);

        // 8. Обновление всех весов
        updateWeights(Wo, dWo);
        updateWeights(Wf, dWf);
        updateWeights(Wi, dWi);
        updateWeights(Wc, dWc); // Добавлено обновление Wc
    }
    
    private void checkNotNull(Object obj, String name) {
        if (obj == null) {
            throw new IllegalStateException(name + " is null");
        }
    }

    private double[] sigmoidGradient(double[] x) {
        double[] s = sigmoid(x);
        return multiply(s, subtract(1.0, s));
    }
    
    private double[] subtract(double scalar, double[] a) {
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) 
            res[i] = scalar - a[i];
        return res;
    }

    private void updateWeights(double[][] W, double[][] grad) {
        for (int i = 0; i < W.length; i++) {
            for (int j = 0; j < W[i].length; j++) {
                W[i][j] -= learningRate * grad[i][j];
            }
        }
    }

    // Vector operations
    private double[] add(double[] a, double[] b) {
        checkDimensions(a, b);
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] + b[i];
        return res;
    }

    private double[] multiply(double[] a, double[] b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("Null array in multiply");
        }
        checkDimensions(a, b);
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] * b[i];
        return res;
    }

    private double[] sigmoid(double[] x) {
        double[] res = new double[x.length];
        for (int i = 0; i < x.length; i++) 
            res[i] = 1.0 / (1.0 + Math.exp(-x[i]));
        return res;
    }

    private double[] tanh(double[] x) {
        double[] res = new double[x.length];
        for (int i = 0; i < x.length; i++) 
            res[i] = Math.tanh(x[i]);
        return res;
    }

    private double[] tanhGradient(double[] x) {
        double[] res = new double[x.length];
        for (int i = 0; i < x.length; i++)
            res[i] = 1 - Math.pow(Math.tanh(x[i]), 2);
        return res;
    }

    private double[] softmax(double[] x) {
        double max = Arrays.stream(x).max().orElse(1.0);
        double[] exp = new double[x.length];
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            exp[i] = Math.exp(x[i] - max);
            sum += exp[i];
        }
        if (sum == 0) Arrays.fill(exp, 1.0 / x.length);
        else for (int i = 0; i < x.length; i++) exp[i] /= sum;
        return exp;
    }

    private double[] subtract(double[] a, double[] b) {
        checkDimensions(a, b);
        double[] res = new double[a.length];
        for (int i = 0; i < a.length; i++) res[i] = a[i] - b[i];
        return res;
    }

    private double[] concatenate(double[] a, double[] b) {
        double[] res = new double[a.length + b.length];
        System.arraycopy(a, 0, res, 0, a.length);
        System.arraycopy(b, 0, res, a.length, b.length);
        return res;
    }

    private double[][] outerProduct(double[] a, double[] b) {
        double[][] res = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++)
            for (int j = 0; j < b.length; j++)
                res[i][j] = a[i] * b[j];
        return res;
    }

    private double[] mvMult(double[][] W, double[] x) {
        double[] res = new double[W.length];
        for (int i = 0; i < W.length; i++)
            for (int j = 0; j < W[i].length; j++)
                res[i] += W[i][j] * x[j];
        return res;
    }

    private double[] mvMultTranspose(double[][] W, double[] x) {
        double[] res = new double[W[0].length];
        for (int j = 0; j < W[0].length; j++)
            for (int i = 0; i < W.length; i++)
                res[j] += W[i][j] * x[i];
        return res;
    }

    private void checkDimensions(double[] a, double[] b) {
        if (a.length != b.length)
            throw new IllegalArgumentException("Dimension mismatch: " + a.length + " vs " + b.length);
    }

    // Training
    public void train(String filePath, int epochs) throws IOException {
        String text = new String(Files.readAllBytes(Paths.get(filePath)));
        String[] tokens = preprocess(text).split(" ");

        // Построение словаря
        for (String token : tokens) {
            if (!token.isEmpty() && !vocab.containsKey(token)) {
                vocab.put(token, vocab.size());
                words.add(token);
            }
        }
        vocabSize = vocab.size();
        initializeWeights();

        // Преобразование в индексы вместо векторов
        List<Integer> inputIndices = new ArrayList<>();
        for (String token : tokens) {
            if (vocab.containsKey(token)) {
                inputIndices.add(vocab.get(token));
            }
        }

        // Цикл обучения с поточной обработкой
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalLoss = 0;
            LSTMState state = new LSTMState(hiddenSize, vocabSize);
            
            for (int i = 0; i < inputIndices.size() - 1; i++) {
                // Создаем one-hot вектор на лету
                double[] inputVec = new double[vocabSize];
                inputVec[inputIndices.get(i)] = 1.0;
                
                // Целевой вектор
                double[] targetVec = new double[vocabSize];
                targetVec[inputIndices.get(i+1)] = 1.0;

                LSTMState newState = lstmStep(inputVec, state);
                totalLoss += crossEntropy(newState.output, targetVec);
                backward(newState, state, targetVec);
                state = newState;

                // Очистка временных данных
                inputVec = null;
                targetVec = null;
            }
            System.out.printf("Epoch %d Loss: %.2f%n", epoch + 1, totalLoss);
        }
    }

    private double crossEntropy(double[] pred, double[] target) {
        double loss = 0;
        for (int i = 0; i < target.length; i++)
            loss += -target[i] * Math.log(pred[i] + 1e-8);
        return loss;
    }

    // Generation
    public String generate(String prompt, int length) {
        String[] tokens = preprocess(prompt).split(" ");
        LSTMState state = new LSTMState(hiddenSize, vocabSize);
        state.h = new double[hiddenSize];
        state.c = new double[hiddenSize];

        for (String token : tokens) {
            if (vocab.containsKey(token)) {
                double[] vec = new double[vocabSize];
                vec[vocab.get(token)] = 1.0;
                state = lstmStep(vec, state);
            }
        }

        StringBuilder output = new StringBuilder();
        for (int i = 0; i < length; i++) {
            int nextIdx = sample(state.output);
            nextIdx = Math.max(0, Math.min(nextIdx, words.size() - 1));
            String nextWord = words.get(nextIdx);
            output.append(" ").append(nextWord);
            double[] vec = new double[vocabSize];
            vec[nextIdx] = 1.0;
            state = lstmStep(vec, state);
        }
        return postprocess(output.toString());
    }

    private int sample(double[] probs) {
        double sum = Arrays.stream(probs).sum();
        if (sum <= 0) return random.nextInt(probs.length);
        double r = random.nextDouble() * sum;
        double accum = 0;
        for (int i = 0; i < probs.length; i++) {
            accum += probs[i];
            if (accum >= r) return i;
        }
        return probs.length - 1;
    }

    // Text processing
    private String preprocess(String text) {
        return text.toLowerCase()
                .replaceAll("[^a-zа-яё0-9'?!., ]", " ")
                .replaceAll("\\s+", " ")
                .trim();
    }

    private String postprocess(String text) {
        return text.replaceAll(" ([.,!?])", "$1")
                .replaceAll("^\\s+", "")
                .replaceAll("\\s+$", "");
    }

    // Chat interface
    public void startChat() {
        System.out.println("Chat Bot Initialized. Type 'exit' to quit.");
        while (true) {
            System.out.print("You: ");
            String input = scanner.nextLine();
            if ("exit".equalsIgnoreCase(input)) break;
            System.out.println("Bot: " + generate(input, 64)); // HERE AN ANSWEAR SETTING!!!
        }
    }
    
 // Сохранение модели в директорию
    public void saveModel(String dirPath) throws IOException {
        File dir = new File(dirPath);
        if (!dir.exists()) dir.mkdirs();

        // Сохраняем веса
        saveMatrix(new File(dir, "Wf.txt"), Wf);
        saveMatrix(new File(dir, "Wi.txt"), Wi);
        saveMatrix(new File(dir, "Wc.txt"), Wc);
        saveMatrix(new File(dir, "Wo.txt"), Wo);
        saveMatrix(new File(dir, "Wy.txt"), Wy);

        // Сохраняем словарь
        try (PrintWriter writer = new PrintWriter(new File(dir, "vocab.txt"))) {
            for (String word : words) {
                writer.println(word);
            }
        }
    }

    // Загрузка модели из директории
    public void loadModel(String dirPath) throws IOException {
        // Загружаем словарь
        words.clear();
        vocab.clear();
        try (BufferedReader reader = new BufferedReader(new FileReader(new File(dirPath, "vocab.txt")))) {
            String line;
            int index = 0;
            while ((line = reader.readLine()) != null) {
                words.add(line);
                vocab.put(line, index++);
            }
        }
        vocabSize = words.size();

        // Загружаем веса
        Wf = loadMatrix(new File(dirPath, "Wf.txt"));
        Wi = loadMatrix(new File(dirPath, "Wi.txt"));
        Wc = loadMatrix(new File(dirPath, "Wc.txt"));
        Wo = loadMatrix(new File(dirPath, "Wo.txt"));
        Wy = loadMatrix(new File(dirPath, "Wy.txt"));
    }

    // Вспомогательные методы для работы с матрицами
    private void saveMatrix(File file, double[][] matrix) throws IOException {
        try (PrintWriter writer = new PrintWriter(file)) {
            for (double[] row : matrix) {
                for (double val : row) {
                    writer.print(val + " ");
                }
                writer.println();
            }
        }
    }

    private double[][] loadMatrix(File file) throws IOException {
        List<double[]> rows = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.trim().split(" ");
                double[] row = new double[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    row[i] = Double.parseDouble(parts[i]);
                }
                rows.add(row);
            }
        }
        return rows.toArray(new double[0][]);
    }
    
    
    
    public static void main(String[] args) throws IOException {
        NeuralChatBot bot = new NeuralChatBot(256, 0.7); // buffer and temperature
        bot.train("chat_corpus.txt", 2); // gens
        bot.saveModel("saved_model");
        bot.startChat();
        
        //NeuralChatBot loadedBot = new NeuralChatBot(1, 0.9);
        //loadedBot.loadModel("saved_model");
        
        //loadedBot.startChat();
    }
}