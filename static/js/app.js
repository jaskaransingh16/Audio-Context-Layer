/* --------------------------------------------------------------------------
   Audio Context Layer - Frontend Client Application with Highlight Visuals
   -------------------------------------------------------------------------- */

document.addEventListener("DOMContentLoaded", () => {
    // 1. Navigation Tab Switching
    const navButtons = document.querySelectorAll(".nav-btn");
    const tabPages = document.querySelectorAll(".tab-page");

    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            navButtons.forEach(b => b.classList.remove("active"));
            tabPages.forEach(p => p.classList.remove("active"));

            btn.classList.add("active");
            const targetId = btn.getAttribute("data-target");
            document.getElementById(targetId).classList.add("active");

            if (targetId === "benchmark-section") {
                loadBenchmarkMetrics();
            }
        });
    });

    // 2. Mode Switching (Dataset Sample vs Upload)
    const btnPreset = document.getElementById("btn-mode-preset");
    const btnUpload = document.getElementById("btn-mode-upload");
    const viewPreset = document.getElementById("preset-container");
    const viewUpload = document.getElementById("upload-container");

    let currentMode = "preset";
    let selectedSampleId = null;
    let uploadedFile = null;

    btnPreset.addEventListener("click", () => {
        currentMode = "preset";
        btnPreset.classList.add("active");
        btnUpload.classList.remove("active");
        viewPreset.classList.add("active");
        viewUpload.classList.remove("active");
    });

    btnUpload.addEventListener("click", () => {
        currentMode = "upload";
        btnUpload.classList.add("active");
        btnPreset.classList.remove("active");
        viewUpload.classList.add("active");
        viewPreset.classList.remove("active");
    });

    // 3. Dataset Sample Selector
    const sampleSelect = document.getElementById("sample-select");
    const metaBox = document.getElementById("sample-metadata-box");
    const metaEnv = document.getElementById("meta-env");
    const metaDur = document.getElementById("meta-dur");
    const metaEvents = document.getElementById("meta-events");
    const audioElement = document.getElementById("audio-element");

    let testSamplesMap = {};

    function fetchTestSamples() {
        fetch("/api/samples")
            .then(res => res.json())
            .then(data => {
                if (data.samples && data.samples.length > 0) {
                    sampleSelect.innerHTML = '<option value="">-- Select Benchmark Sample --</option>';
                    data.samples.forEach(s => {
                        testSamplesMap[s.sample_id] = s;
                        const opt = document.createElement("option");
                        opt.value = s.sample_id;
                        opt.textContent = `${s.sample_id} • ${s.environment.toUpperCase()} (${s.duration}s, ${s.event_count} events)`;
                        sampleSelect.appendChild(opt);
                    });

                    sampleSelect.selectedIndex = 1;
                    onSampleSelectChange();
                }
            })
            .catch(err => console.error("Error loading samples:", err));
    }

    function onSampleSelectChange() {
        const sid = sampleSelect.value;
        if (sid && testSamplesMap[sid]) {
            selectedSampleId = sid;
            const sample = testSamplesMap[sid];

            metaBox.classList.remove("hidden");
            metaEnv.textContent = sample.environment.toUpperCase();
            metaDur.textContent = `${sample.duration}s`;
            
            const eventNames = sample.events.map(e => e.class);
            metaEvents.textContent = eventNames.length > 0 ? [...new Set(eventNames)].join(", ") : "none";

            audioElement.src = `/audio/${sample.audio_file}`;
            audioElement.load();
            drawWaveformPlaceholder();
        } else {
            metaBox.classList.add("hidden");
            selectedSampleId = null;
        }
    }

    sampleSelect.addEventListener("change", onSampleSelectChange);
    fetchTestSamples();

    // 4. File Upload Drag & Drop
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("file-input");
    const fileNameDisplay = document.getElementById("file-name-display");

    dropzone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "#38bdf8";
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.style.borderColor = "#27272a";
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "#27272a";
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    function handleFile(file) {
        uploadedFile = file;
        fileNameDisplay.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
        const fileUrl = URL.createObjectURL(file);
        audioElement.src = fileUrl;
        audioElement.load();
        drawWaveformPlaceholder();
    }

    // 5. Query Suggestion Chips
    const questionInput = document.getElementById("question-input");
    const chipBtns = document.querySelectorAll(".chip-btn");

    chipBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const promptText = btn.getAttribute("data-q");
            questionInput.value = promptText;
        });
    });

    // 6. Chart.js Attention Heatmap Chart with Color Highlighting
    let attentionChart = null;

    function initAttentionChart(timestamps, weights) {
        const ctx = document.getElementById("attention-chart").getContext("2d");
        if (attentionChart) {
            attentionChart.destroy();
        }

        // Color highlighting depending on attention weight magnitude
        const bgColors = weights.map(w => {
            if (w >= 0.6) return "rgba(56, 189, 248, 0.95)";  // High attention: Bright Cyan
            if (w >= 0.3) return "rgba(59, 130, 246, 0.75)";  // Medium attention: Indigo
            return "rgba(59, 130, 246, 0.25)";                 // Low attention: Subtle Blue
        });

        const borderColors = weights.map(w => {
            if (w >= 0.6) return "#38bdf8";
            if (w >= 0.3) return "#3b82f6";
            return "rgba(59, 130, 246, 0.4)";
        });

        attentionChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: timestamps.map(t => `${t}s`),
                datasets: [{
                    label: "Cross-Attention α(t)",
                    data: weights,
                    backgroundColor: bgColors,
                    borderColor: borderColors,
                    borderWidth: 1,
                    borderRadius: 3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: "#94a3b8", font: { size: 10, family: "Inter" } }
                    },
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        grid: { color: "#1f1f23" },
                        ticks: { color: "#94a3b8", font: { size: 10, family: "Inter" } }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    // 7. Mel Spectrogram Canvas Renderer with Viridis Highlighting
    function drawSpectrogramCanvas(matrix) {
        const canvas = document.getElementById("spectrogram-canvas");
        const ctx = canvas.getContext("2d");
        
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = canvas.parentElement.clientHeight;

        if (!matrix || matrix.length === 0) return;

        const numMels = matrix.length;
        const numFrames = matrix[0].length;

        const cellWidth = canvas.width / numFrames;
        const cellHeight = canvas.height / numMels;

        let minVal = Infinity, maxVal = -Infinity;
        for (let r = 0; r < numMels; r++) {
            for (let c = 0; c < numFrames; c++) {
                if (matrix[r][c] < minVal) minVal = matrix[r][c];
                if (matrix[r][c] > maxVal) maxVal = matrix[r][c];
            }
        }

        const range = maxVal - minVal + 1e-6;

        for (let r = 0; r < numMels; r++) {
            for (let c = 0; c < numFrames; c++) {
                const norm = (matrix[r][c] - minVal) / range;
                
                // Vivid Viridis Spectral Heat Color Palette
                let red, green, blue;
                if (norm < 0.25) {
                    // Dark Indigo to Purple
                    red = Math.floor(norm * 4 * 68);
                    green = Math.floor(norm * 4 * 1);
                    blue = Math.floor(84 + norm * 4 * (140 - 84));
                } else if (norm < 0.5) {
                    // Teal to Emerald
                    const n = (norm - 0.25) * 4;
                    red = Math.floor(33 + n * (33 - 33));
                    green = Math.floor(100 + n * (175 - 100));
                    blue = Math.floor(140 + n * (133 - 140));
                } else if (norm < 0.75) {
                    // Emerald to Amber
                    const n = (norm - 0.5) * 4;
                    red = Math.floor(33 + n * (245 - 33));
                    green = Math.floor(175 + n * (190 - 175));
                    blue = Math.floor(133 - n * 100);
                } else {
                    // Amber to Bright Yellow Peak
                    const n = (norm - 0.75) * 4;
                    red = Math.floor(245 + n * 10);
                    green = Math.floor(190 + n * 50);
                    blue = Math.floor(33 + n * 180);
                }

                ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
                ctx.fillRect(c * cellWidth, (numMels - 1 - r) * cellHeight, cellWidth + 0.5, cellHeight + 0.5);
            }
        }
    }

    function drawWaveformPlaceholder() {
        const canvas = document.getElementById("waveform-canvas");
        const ctx = canvas.getContext("2d");
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = canvas.parentElement.clientHeight;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        const midY = canvas.height / 2;

        // Draw waveform filled area with cyan highlight gradient
        const grad = ctx.createLinearGradient(0, 0, 0, canvas.height);
        grad.addColorStop(0, "rgba(56, 189, 248, 0.25)");
        grad.addColorStop(0.5, "rgba(56, 189, 248, 0.05)");
        grad.addColorStop(1, "rgba(56, 189, 248, 0.25)");

        ctx.fillStyle = grad;
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 1.5;

        ctx.beginPath();
        ctx.moveTo(0, midY);

        for (let x = 0; x < canvas.width; x += 3) {
            const amp = Math.sin(x * 0.05) * Math.cos(x * 0.02) * (canvas.height * 0.35);
            ctx.lineTo(x, midY + amp);
        }

        ctx.lineTo(canvas.width, midY);
        ctx.stroke();
    }

    // 8. Inference Execution
    const btnSubmit = document.getElementById("btn-submit-qa");
    const statusBadge = document.getElementById("status-badge");
    const answerText = document.getElementById("answer-text");
    const catBadge = document.getElementById("cat-badge");

    btnSubmit.addEventListener("click", () => {
        const q = questionInput.value.trim();
        if (!q) {
            alert("Please enter a question!");
            return;
        }

        const formData = new FormData();
        formData.append("question", q);

        if (currentMode === "preset") {
            if (!selectedSampleId) {
                alert("Please select a benchmark sample!");
                return;
            }
            formData.append("sample_id", selectedSampleId);
        } else {
            if (!uploadedFile) {
                alert("Please select a WAV file to upload!");
                return;
            }
            formData.append("audio_file", uploadedFile);
        }

        statusBadge.textContent = "Processing...";
        statusBadge.className = "status-badge thinking";
        answerText.textContent = "Computing context answer...";

        fetch("/api/predict", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.status === "success" && data.result) {
                const res = data.result;
                answerText.textContent = res.answer;
                
                const catLower = res.question_category.toLowerCase();
                catBadge.textContent = res.question_category.toUpperCase();
                catBadge.className = `cat-badge ${catLower}`;

                statusBadge.textContent = "Ready";
                statusBadge.className = "status-badge ready";

                initAttentionChart(res.temporal_attention.timestamps, res.temporal_attention.weights);
                drawSpectrogramCanvas(res.spectrogram.matrix);
            } else {
                answerText.textContent = "Error: " + (data.error || "Failed to process");
                statusBadge.textContent = "Error";
            }
        })
        .catch(err => {
            console.error("Prediction Error:", err);
            answerText.textContent = "Error connecting to server";
            statusBadge.textContent = "Error";
        });
    });

    // 9. Benchmark Metrics Loader
    let breakdownChart = null;

    function loadBenchmarkMetrics() {
        fetch("/api/benchmark_metrics")
            .then(res => res.json())
            .then(data => {
                if (data.overall) {
                    document.getElementById("metric-em").textContent = `${data.overall.exact_match_accuracy}%`;
                    document.getElementById("metric-bleu1").textContent = data.overall.bleu1;
                    document.getElementById("metric-bleu4").textContent = data.overall.bleu4;
                    document.getElementById("metric-rouge").textContent = data.overall.rouge_l;
                }

                if (data.category_breakdown) {
                    const cats = Object.keys(data.category_breakdown);
                    const accs = cats.map(c => data.category_breakdown[c].accuracy);

                    const tbody = document.getElementById("benchmark-table-body");
                    tbody.innerHTML = "";

                    const catColors = {
                        perceptual: "#38bdf8",
                        counting: "#34d399",
                        temporal: "#fbbf24",
                        causal: "#c084fc"
                    };

                    cats.forEach(c => {
                        const item = data.category_breakdown[c];
                        const tr = document.createElement("tr");
                        const color = catColors[c.toLowerCase()] || "#38bdf8";
                        tr.innerHTML = `
                            <td><strong style="color:${color}">${c.toUpperCase()}</strong></td>
                            <td>${item.sample_count}</td>
                            <td><span style="color:#34d399; font-weight:600;">${item.accuracy}%</span></td>
                            <td>${item.bleu1}</td>
                            <td>${item.rouge_l}</td>
                        `;
                        tbody.appendChild(tr);
                    });

                    const ctx = document.getElementById("breakdown-chart").getContext("2d");
                    if (breakdownChart) breakdownChart.destroy();

                    breakdownChart = new Chart(ctx, {
                        type: "bar",
                        data: {
                            labels: cats.map(c => c.toUpperCase()),
                            datasets: [{
                                label: "Exact Match Accuracy (%)",
                                data: accs,
                                backgroundColor: ["#38bdf8", "#34d399", "#fbbf24", "#c084fc"],
                                borderRadius: 4
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: { beginAtZero: true, max: 100, ticks: { color: "#94a3b8", font: { family: "Inter" } }, grid: { color: "#1f1f23" } },
                                x: { ticks: { color: "#94a3b8", font: { family: "Inter" } }, grid: { display: false } }
                            },
                            plugins: { legend: { display: false } }
                        }
                    });
                }
            })
            .catch(err => console.error("Error loading benchmark metrics:", err));
    }
});
