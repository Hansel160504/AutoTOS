// static/js/create_tos.js
document.addEventListener("DOMContentLoaded", () => {

    // ── DOM refs ──
    const addRowBtn           = document.getElementById("addRow");
    const generateBtn         = document.getElementById("generateTOSQuizBtn");
    const tosTable            = document.getElementById("tosTable");
    const outputBlock         = document.getElementById("generatedOutput");
    const previewArea         = document.getElementById('quiz-preview-area');
    const previewBody         = document.getElementById('preview-body');
    const subjectTypeSelect   = document.getElementById("subjectType");
    const customPercentBlock  = document.getElementById("customPercentBlock");
    const famPercentInput     = document.getElementById("famPercentInput");
    const intPercentInput     = document.getElementById("intPercentInput");
    const crePercentInput     = document.getElementById("crePercentInput");
    const percentValidationMsg= document.getElementById("percentValidationMsg");
    const testModal           = document.getElementById("testModal");
    const addTestBtn          = document.getElementById("addTestBtn");
    const confirmTestsBtn     = document.getElementById("confirmTestsBtn");
    const cancelTestModalBtn  = document.getElementById("cancelTestModal");
    const testList            = document.getElementById("testList");
    const testTotalCount      = document.getElementById("testTotalCount");
    const loadingOverlay      = document.getElementById("loadingOverlay");
    const cancelGenerationBtn = document.getElementById("cancelGenerationBtn");
    const staticCloseX        = document.getElementById('previewCloseX');
    const staticCloseBtn      = document.getElementById('quiz-preview-close');
    const staticSaveBtn       = document.getElementById('quiz-preview-save');

    // ── Progress indicator elements ──
    const spinnerCount    = document.getElementById('spinnerCount');
    const loadingStatus   = document.getElementById('loadingStatus');
    const progressBarFill = document.getElementById('progressBarFill');
    const stepAnalyze     = document.getElementById('stepAnalyze');
    const stepGenerate    = document.getElementById('stepGenerate');
    const stepFinalize    = document.getElementById('stepFinalize');

    if (!addRowBtn || !generateBtn || !tosTable) {
        console.error("TOS: required elements missing.");
        return;
    }

    let tests           = [];
    let abortController = null;
    let currentMasterId = null;
    let progressInterval = null;  // tracks the counter timer

    if (percentValidationMsg) percentValidationMsg.style.display = "none";

    // ================================================================
    // PROGRESS INDICATOR
    // ================================================================
    function startProgress(totalItems) {
        let current = 0;
        const total = totalItems || 10;

        // Reset UI
        if (spinnerCount)    spinnerCount.textContent  = '0';
        if (loadingStatus)   loadingStatus.textContent = 'Analyzing materials…';
        if (progressBarFill) progressBarFill.style.width = '0%';

        setStep('analyze');

        // Phase timing:
        // 0–20% → Analyze (fast)
        // 20–90% → Generate items one by one
        // 90–100% → Finalize (fast)

        const analyzeMs  = 1200;                        // 1.2s for analyze phase
        const generateMs = Math.max(800, 80000 / total); // ~1s per item, capped
        const finalizeMs = 800;

        let phase = 'analyze';
        let analyzeTimer = null;

        // ── Phase 1: Analyze ──
        let analyzeProgress = 0;
        analyzeTimer = setInterval(() => {
            analyzeProgress += 2;
            const pct = Math.round((analyzeProgress / 100) * 20);  // 0→20%
            setBar(pct);
            if (analyzeProgress >= 100) {
                clearInterval(analyzeTimer);
                phase = 'generate';
                setStep('generate');
                startGeneratePhase();
            }
        }, analyzeMs / 50);

        // ── Phase 2: Generate items ──
        function startGeneratePhase() {
            let itemsDone = 0;
            progressInterval = setInterval(() => {
                itemsDone++;
                const pct = 20 + Math.round((itemsDone / total) * 70); // 20→90%
                current = itemsDone;

                if (spinnerCount)  spinnerCount.textContent  = current;
                if (loadingStatus) loadingStatus.textContent =
                    `Generating item ${current} of ${total}…`;
                setBar(Math.min(pct, 90));

                if (itemsDone >= total) {
                    clearInterval(progressInterval);
                    progressInterval = null;
                    startFinalizePhase();
                }
            }, generateMs);
        }

        // ── Phase 3: Finalize ──
        function startFinalizePhase() {
            setStep('finalize');
            if (loadingStatus) loadingStatus.textContent = 'Finalizing quiz…';

            let fp = 90;
            const finalTimer = setInterval(() => {
                fp += 2;
                setBar(Math.min(fp, 99));
                if (fp >= 99) clearInterval(finalTimer);
            }, finalizeMs / 5);
        }
    }

    function stopProgress() {
        if (progressInterval) { clearInterval(progressInterval); progressInterval = null; }
        // Complete the bar
        if (progressBarFill) progressBarFill.style.width = '100%';
        if (spinnerCount)    spinnerCount.textContent  = '✓';
        if (loadingStatus)   loadingStatus.textContent = 'Done!';
        setStep('finalize', true);
    }

    function resetProgress() {
        if (progressInterval) { clearInterval(progressInterval); progressInterval = null; }
        if (spinnerCount)    spinnerCount.textContent  = '0';
        if (loadingStatus)   loadingStatus.textContent = 'Starting up…';
        if (progressBarFill) progressBarFill.style.width = '0%';
        setStep('analyze');
    }

    function setBar(pct) {
        if (progressBarFill) progressBarFill.style.width = pct + '%';
    }

    function setStep(active, allDone) {
        const steps = { analyze: stepAnalyze, generate: stepGenerate, finalize: stepFinalize };
        const order = ['analyze', 'generate', 'finalize'];
        const activeIdx = order.indexOf(active);
        order.forEach((key, i) => {
            const el = steps[key];
            if (!el) return;
            el.classList.remove('active', 'done');
            if (allDone) {
                el.classList.add('done');
            } else if (i < activeIdx) {
                el.classList.add('done');
            } else if (i === activeIdx) {
                el.classList.add('active');
            }
        });
    }

    // ================================================================
    // HELPER: hide/show preview overlay
    // ================================================================
    function hidePreview() {
        if (!previewArea) return;
        previewArea.style.display = 'none';
        previewArea.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
    }

    function showPreview() {
        if (!previewArea) return;
        previewArea.style.display = 'flex';
        previewArea.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
        setTimeout(() => {
            const sheet = previewArea.querySelector('.bondpaper');
            if (sheet) sheet.scrollTop = 0;
        }, 40);
    }

    // ================================================================
    // FRAGMENT CHECKBOX UTILITIES
    // ================================================================
    function getFragmentCBs() {
        return Array.from(document.querySelectorAll('#quiz-preview-card .question-select-cb'));
    }

    function highlightFragmentItem(cb) {
        const li = cb.closest('li');
        if (li) li.classList.toggle('q-selected', cb.checked);
    }

    function updateFragmentCount() {
        const all     = getFragmentCBs();
        const checked = all.filter(cb => cb.checked);
        const countEl = document.getElementById('preview-selected-count');
        const saveBtn = document.getElementById('quiz-preview-save-selected');
        if (countEl) {
            countEl.textContent = checked.length > 0
                ? `${checked.length} / ${all.length} selected`
                : '0 selected';
        }
        if (saveBtn) saveBtn.disabled = checked.length === 0;
    }

    function setAllFragmentCBs(state) {
        getFragmentCBs().forEach(cb => {
            cb.checked = state;
            highlightFragmentItem(cb);
        });
        updateFragmentCount();
    }

    // ================================================================
    // STATIC BUTTON LISTENERS
    // ================================================================
    if (staticCloseX)   staticCloseX.addEventListener('click', hidePreview);
    if (staticCloseBtn) staticCloseBtn.addEventListener('click', hidePreview);
    if (staticSaveBtn) {
        staticSaveBtn.addEventListener('click', () => {
            const r = document.querySelector('input[name="redirect_after_save"]');
            if (r && r.value) window.location = r.value;
            else hidePreview();
        });
    }

    // ================================================================
    // EVENT DELEGATION ON previewArea
    // ================================================================
    if (previewArea) {
        previewArea.addEventListener('click', function (ev) {
            const id = ev.target.id;
            if (id === 'quiz-preview-close') { hidePreview(); return; }
            if (id === 'quiz-preview-save') {
                const r = document.querySelector('input[name="redirect_after_save"]');
                if (r && r.value) window.location = r.value;
                else hidePreview();
                return;
            }
            if (id === 'preview-select-all-btn')   { setAllFragmentCBs(true);  return; }
            if (id === 'preview-deselect-all-btn')  { setAllFragmentCBs(false); return; }

            if (id === 'quiz-preview-save-selected') {
                ev.preventDefault(); ev.stopPropagation();
                const checked = getFragmentCBs().filter(cb => cb.checked);
                if (!checked.length) return;
                const indices = checked.map(cb => parseInt(cb.dataset.qIndex, 10));
                const btn = ev.target;
                btn.textContent = 'Saving…'; btn.disabled = true; btn.style.background = '#d97706';
                callSaveSelected(indices,
                    (result) => {
                        btn.textContent = `✓ Saved ${result.total_items} items!`;
                        btn.style.background = '#16a34a';
                        setTimeout(() => {
                            const r = document.querySelector('input[name="redirect_after_save"]');
                            if (r && r.value) window.location.href = r.value;
                            else hidePreview();
                        }, 900);
                    },
                    (msg) => {
                        alert('Save Selected failed: ' + msg);
                        btn.textContent = 'Save Selected'; btn.style.background = '#d97706'; btn.disabled = false;
                    }
                );
                return;
            }
        });

        previewArea.addEventListener('change', function (ev) {
            if (ev.target && ev.target.classList.contains('question-select-cb')) {
                highlightFragmentItem(ev.target);
                updateFragmentCount();
            }
        });
    }

    // ================================================================
    // MutationObserver
    // ================================================================
    const obs = new MutationObserver((mutations) => {
        for (const m of mutations) {
            if (m.type === 'childList' && previewBody && previewBody.children.length > 0) {
                showPreview();
                setTimeout(updateFragmentCount, 0);
                break;
            }
        }
    });
    if (previewBody) obs.observe(previewBody, { childList: true, subtree: false });

    // ================================================================
    // callSaveSelected
    // ================================================================
    async function callSaveSelected(selectedIndices, onSuccess, onError) {
        if (!currentMasterId) { onError("Master record ID is missing. Please regenerate the quiz."); return; }
        if (!selectedIndices || selectedIndices.length === 0) { onError("No questions selected."); return; }
        try {
            const resp = await fetch("/dashboard/save_selected", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ parent_id: currentMasterId, selected_indices: selectedIndices }),
            });
            const result = await resp.json();
            if (!resp.ok || result.error) { onError(result.error || `Server error: ${resp.status}`); return; }
            onSuccess(result);
        } catch (err) {
            console.error("save_selected error:", err);
            onError("Network error. Please try again.");
        }
    }

    // ================================================================
    // HELPERS
    // ================================================================
    function _injectMasterIdInput(masterId) {
        if (!masterId) return;
        let el = document.getElementById('_master_id_input');
        if (!el) {
            el = document.createElement('input');
            el.type = 'hidden'; el.id = '_master_id_input'; el.name = 'master_id';
            document.body.appendChild(el);
        }
        el.value = masterId;
    }

    function _setRedirectInput(url) {
        let el = document.querySelector('input[name="redirect_after_save"]');
        if (!el) {
            el = document.createElement('input');
            el.type = 'hidden'; el.name = 'redirect_after_save';
            document.body.appendChild(el);
        }
        el.value = url;
    }

    // ================================================================
    // CUSTOM PERCENT BLOCK
    // ================================================================
    function updateCustomUI() {
        if (!subjectTypeSelect || !customPercentBlock) return;
        const show = subjectTypeSelect.value === 'custom';
        customPercentBlock.style.display = show ? 'block' : 'none';
        customPercentBlock.setAttribute('aria-hidden', show ? 'false' : 'true');
        if (!show && percentValidationMsg) percentValidationMsg.style.display = 'none';
    }
    if (subjectTypeSelect) { subjectTypeSelect.addEventListener('change', updateCustomUI); updateCustomUI(); }

    // ================================================================
    // TOPIC TABLE
    // ================================================================
    addRowBtn.onclick = () => {
        const tbody = tosTable.querySelector("tbody");
        const row   = document.createElement("tr");
        row.innerHTML = `
            <td><input type="text" placeholder="e.g. Deep Learning"></td>
            <td><input type="number" value="3" min="1"></td>
            <td><input type="file" class="learnFile"></td>
            <td><button type="button" class="btn secondary remove">Remove</button></td>
        `;
        tbody.appendChild(row);
    };

    tosTable.addEventListener("click", (e) => {
        if (e.target.classList.contains("remove")) e.target.closest("tr").remove();
    });

    const toBase64 = (file) => new Promise((res, rej) => {
        const r = new FileReader();
        r.readAsDataURL(file);
        r.onload  = () => res(r.result);
        r.onerror = (err) => rej(err);
    });

    // ================================================================
    // TEST MODAL
    // ================================================================
    function updateTestCount() {
        let total = 0;
        document.querySelectorAll(".testItems").forEach(i => { total += parseInt(i.value || 0); });
        if (testTotalCount) testTotalCount.textContent = total;
    }

    generateBtn.onclick = () => {
        const title     = document.getElementById("tosTitle").value.trim();
        const totalQuiz = parseInt(document.getElementById("totalQuizItemsInput").value, 10);

        if (!title)                       { alert("Enter TOS title."); return; }
        if (!totalQuiz || totalQuiz <= 0) { alert("Enter a valid total number of quiz items."); return; }

        if (subjectTypeSelect && subjectTypeSelect.value === "custom") {
            const fam  = parseInt(famPercentInput.value  || 0, 10);
            const inti = parseInt(intPercentInput.value  || 0, 10);
            const cre  = parseInt(crePercentInput.value  || 0, 10);
            if (fam < 0 || inti < 0 || cre < 0 || fam > 100 || inti > 100 || cre > 100) {
                if (percentValidationMsg) { percentValidationMsg.textContent = "Each percentage must be between 0 and 100."; percentValidationMsg.style.display = "block"; }
                return;
            }
            if (fam + inti + cre !== 100) {
                if (percentValidationMsg) { percentValidationMsg.textContent = "Percentages must sum to exactly 100."; percentValidationMsg.style.display = "block"; }
                return;
            }
            if (percentValidationMsg) percentValidationMsg.style.display = "none";
        }

        if (testModal) testModal.style.display = "flex";
        tests = [];
        if (testList) testList.innerHTML = "";
        if (testTotalCount) testTotalCount.textContent = 0;
        if (addTestBtn) addTestBtn.click();
    };

    if (addTestBtn) {
        addTestBtn.onclick = () => {
            const div = document.createElement("div");
            div.classList.add("test-row");
            div.innerHTML = `
                <select class="testType">
                    <option value="mcq">Multiple Choice</option>
                    <option value="truefalse">True/False</option>
                    <option value="open_ended">Open-Ended</option>
                </select>
                <input type="number" class="testItems items-input" value="5" min="1">
                <input type="text"   class="testDesc desc-input" placeholder="Short description / instruction">
                <button class="removeTest btn secondary" type="button">✕</button>
            `;
            if (testList) testList.appendChild(div);
            updateTestCount();
        };
    }

    document.addEventListener("click",  (e) => { if (e.target.classList.contains("removeTest")) { e.target.closest(".test-row").remove(); updateTestCount(); } });
    document.addEventListener("input",  (e) => { if (e.target.classList.contains("testItems")) updateTestCount(); });

    if (cancelTestModalBtn) cancelTestModalBtn.onclick = () => { if (testModal) testModal.style.display = "none"; };

    // ================================================================
    // CANCEL GENERATION
    // ================================================================
    if (cancelGenerationBtn) {
        cancelGenerationBtn.onclick = () => {
            if (abortController) { abortController.abort(); abortController = null; }
            resetProgress();
            if (loadingOverlay) loadingOverlay.style.display = 'none';
            alert("Generation cancelled.");
        };
    }

    // ================================================================
    // CONFIRM & SUBMIT
    // ================================================================
    if (confirmTestsBtn) {
        confirmTestsBtn.onclick = () => {
            const totalQuiz  = parseInt(document.getElementById("totalQuizItemsInput").value, 10);
            const totalTests = parseInt(testTotalCount ? testTotalCount.textContent : '0', 10);
            if (totalTests !== totalQuiz) { alert(`Test items (${totalTests}) must equal quiz items (${totalQuiz}).`); return; }

            tests = [];
            document.querySelectorAll(".test-row").forEach(row => {
                tests.push({
                    type:        row.querySelector(".testType").value,
                    items:       parseInt(row.querySelector(".testItems").value),
                    description: row.querySelector(".testDesc").value.trim(),
                });
            });

            if (testModal) testModal.style.display = "none";

            // Show overlay and START progress counter
            const totalQuizItems = parseInt(document.getElementById("totalQuizItemsInput").value, 10);
            resetProgress();
            if (loadingOverlay) loadingOverlay.style.display = 'flex';
            startProgress(totalQuizItems);

            submitTOSWithTests(tests);
        };
    }

    // ================================================================
    // SUBMIT
    // ================================================================
    async function submitTOSWithTests(tests) {
        abortController = new AbortController();
        const signal    = abortController.signal;

        const title       = document.getElementById("tosTitle").value.trim();
        const subjectType = subjectTypeSelect ? subjectTypeSelect.value : 'nonlab';
        const totalQuiz   = parseInt(document.getElementById("totalQuizItemsInput").value, 10);

        const topics = [];
        for (const row of Array.from(tosTable.querySelectorAll("tbody tr"))) {
            const inputs     = row.querySelectorAll("input");
            const topicName  = inputs[0] ? inputs[0].value.trim() : '';
            const hoursValue = parseInt(inputs[1] ? inputs[1].value : '0', 10);
            const fileInput  = inputs[2];
            if (!topicName || !hoursValue || hoursValue <= 0) continue;
            let b64 = null;
            if (fileInput && fileInput.files && fileInput.files[0]) {
                try { b64 = await toBase64(fileInput.files[0]); } catch (e) { console.error(e); }
            }
            topics.push({ topic: topicName, hours: hoursValue, learn_material: b64 });
        }

        if (topics.length === 0) {
            alert("Add at least 1 valid topic.");
            resetProgress();
            if (loadingOverlay) loadingOverlay.style.display = 'none';
            return;
        }

        const payload = { title, subjectType, totalQuizItems: totalQuiz, topics, tests };
        if (subjectType === "custom") {
            payload.fam_pct = parseInt(famPercentInput.value || 0, 10);
            payload.int_pct = parseInt(intPercentInput.value || 0, 10);
            payload.cre_pct = parseInt(crePercentInput.value || 0, 10);
        }

        try {
            const resp = await fetch("/dashboard/save_tos", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
                signal,
            });

            // Stop and complete the progress bar
            stopProgress();
            if (loadingOverlay) loadingOverlay.style.display = 'none';

            if (!resp.ok) {
                const txt = await resp.text();
                try { alert(JSON.parse(txt).error || `Error ${resp.status}`); }
                catch (e) { alert(`Error ${resp.status}: ${txt.slice(0, 200)}`); }
                return;
            }

            const data = await resp.json();
            if (data.error) { alert(data.error); return; }

            if (data.master_id) currentMasterId = data.master_id;

            // PATH A: server returned rendered preview HTML
            if (data.preview_html) {
                _injectMasterIdInput(data.master_id);
                if (data.redirect_url) _setRedirectInput(data.redirect_url);
                if (previewBody) {
                    previewBody.innerHTML = '';
                    previewBody.insertAdjacentHTML('beforeend', data.preview_html);
                } else {
                    previewArea.innerHTML =
                        `<div class="bondpaper"><div class="preview-body" id="preview-body">${data.preview_html}</div></div>`;
                    showPreview();
                    setTimeout(updateFragmentCount, 0);
                }
                return;
            }

            if (data.redirect_url && !data.quizzes) { window.location = data.redirect_url; return; }

            // PATH B: fallback JSON rendering
            if (data.quizzes || data.topics) {
                renderTOS(data);
                renderQuiz(data);
                if (outputBlock) outputBlock.style.display = "block";
                showPreview();
            } else {
                alert("Generation completed but no preview was returned.");
            }

        } catch (err) {
            resetProgress();
            if (loadingOverlay) loadingOverlay.style.display = 'none';
            if (err.name === 'AbortError') console.log("Fetch aborted.");
            else { console.error(err); alert("An error occurred. See console."); }
        } finally {
            abortController = null;
        }
    }

    // ================================================================
    // RENDER FUNCTIONS (PATH B fallback)
    // ================================================================
    function renderTOS(data) {
        const body         = document.getElementById("generatedBody");
        const famPercent   = document.getElementById("famPercent");
        const intPercent   = document.getElementById("intPercent");
        const crePercent   = document.getElementById("crePercent");
        const footTotalHrs = document.getElementById("footTotalHrs");
        const footFamItems = document.getElementById("footFamItems");
        const footIntItems = document.getElementById("footIntItems");
        const footCreItems = document.getElementById("footCreItems");
        const footTotal    = document.getElementById("footTotal");
        if (!body) return;

        body.innerHTML = "";
        if (famPercent && data.fam_pct != null) famPercent.textContent = data.fam_pct + "%";
        if (intPercent && data.int_pct != null) intPercent.textContent = data.int_pct + "%";
        if (crePercent && data.cre_pct != null) crePercent.textContent = data.cre_pct + "%";

        let tH=0, tF=0, tI=0, tC=0, tT=0;
        (data.topics || []).forEach(t => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td>${t.topic}</td><td>${t.hours}</td>
                <td>${t.fam??0}</td><td>${t.fam_range??""}</td>
                <td>${t.int??0}</td><td>${t.int_range??""}</td>
                <td>${t.cre??0}</td><td>${t.cre_range??""}</td>
                <td>${t.items??t.quiz_items??0}</td>`;
            body.appendChild(tr);
            tH += t.hours||0; tF += t.fam||0; tI += t.int||0; tC += t.cre||0; tT += t.items||t.quiz_items||0;
        });
        if (footTotalHrs) footTotalHrs.textContent = tH;
        if (footFamItems) footFamItems.textContent = tF;
        if (footIntItems) footIntItems.textContent = tI;
        if (footCreItems) footCreItems.textContent = tC;
        if (footTotal)    footTotal.textContent    = tT;
    }

    function renderQuiz(data) {
        const quizArea = document.getElementById("quizArea");
        if (!quizArea) return;
        quizArea.innerHTML = "";

        const quizzes = data.quizzes || [];
        let curHeader = "";

        const toolbar = document.createElement("div");
        toolbar.innerHTML = `
            <div style="display:flex;align-items:center;gap:8px;margin:0 0 14px;padding:8px 12px;background:#f1f5f9;border:1px solid #e2e8f0;border-radius:8px;font-size:13px;">
                <button id="quiz-area-select-all" type="button" style="background:#fff;border:1.5px solid #e2e8f0;border-radius:7px;padding:5px 12px;cursor:pointer;font-size:12px;font-weight:600;color:#1a2e4a;">☑ Select All</button>
                <button id="quiz-area-deselect-all" type="button" style="background:#fff;border:1.5px solid #e2e8f0;border-radius:7px;padding:5px 12px;cursor:pointer;font-size:12px;font-weight:600;color:#1a2e4a;">☐ Deselect All</button>
                <span id="quiz-area-count" style="margin-left:auto;color:#2563eb;font-weight:700;font-size:13px;">0 selected</span>
            </div>`;
        quizArea.appendChild(toolbar);

        quizzes.forEach((q, idx) => {
            if (q.test_header && q.test_header !== curHeader) {
                const hd = document.createElement("div");
                hd.innerHTML = `<h3 style="color:#2563eb;margin-top:28px;border-bottom:2px solid #e2e8f0;padding-bottom:6px;font-family:'DM Serif Display',serif;">${q.test_header}</h3>`;
                if (q.test_description) hd.innerHTML += `<p style="font-size:13px;color:#64748b;font-style:italic;margin-bottom:14px;">${q.test_description}</p>`;
                quizArea.appendChild(hd);
                curHeader = q.test_header;
            }

            const wrap = document.createElement("div");
            wrap.className = "quiz-item"; wrap.dataset.qIndex = idx + 1;
            wrap.style.cssText = "margin-bottom:18px;padding:14px;border:1px solid #e2e8f0;border-radius:8px;background:#fff;display:flex;gap:12px;align-items:flex-start;transition:background 0.15s,box-shadow 0.15s;";

            const cb = document.createElement("input");
            cb.type = "checkbox"; cb.className = "quiz-area-cb"; cb.dataset.qIndex = idx + 1;
            cb.style.cssText = "margin-top:3px;flex-shrink:0;width:16px;height:16px;cursor:pointer;accent-color:#2563eb;";
            cb.addEventListener("change", function () {
                wrap.style.background = this.checked ? "#eff6ff" : "#fff";
                wrap.style.boxShadow  = this.checked ? "inset 3px 0 0 #2563eb" : "none";
                updateFallbackCount();
            });

            let choicesHTML = "";
            let displayAnswer = q.answer || '';
            if (Array.isArray(q.choices) && q.choices.length) {
                choicesHTML = '<ul style="list-style:none;padding:0;margin:8px 0;">';
                q.choices.forEach((c, i) => {
                    const letter = String.fromCharCode(97 + i);
                    choicesHTML += `<li style="padding:6px 10px;margin-bottom:4px;border-radius:6px;background:#f8fafc;border:1px solid #e2e8f0;font-size:13px;">${letter}) ${c}</li>`;
                    if (displayAnswer.toLowerCase() === letter || displayAnswer.toLowerCase() === c.toLowerCase())
                        displayAnswer = `${letter}) ${c}`;
                });
                choicesHTML += '</ul>';
            }

            const content = document.createElement("div");
            content.style.cssText = "flex:1;min-width:0;";
            content.innerHTML = `
                <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.4px;color:#94a3b8;margin-bottom:5px;">
                    Q${idx+1} · ${(q.type||'').toUpperCase()} · ${q.bloom||''} · ${q.concept||''}
                </div>
                <div style="font-size:14px;font-weight:500;color:#0f172a;margin-bottom:10px;">${q.question||''}</div>
                ${choicesHTML}
                <div style="font-size:13px;background:#f0fdf4;padding:10px 14px;border-left:3px solid #16a34a;border-radius:6px;margin-top:8px;">
                    <strong>Answer:</strong> ${displayAnswer}
                    ${q.answer_text ? `<div style="font-size:12px;color:#64748b;margin-top:4px;">${q.answer_text}</div>` : ''}
                </div>`;

            wrap.appendChild(cb);
            wrap.appendChild(content);
            quizArea.appendChild(wrap);
        });

        if (quizzes.length > 0) {
            const actDiv = document.createElement("div");
            actDiv.style.cssText = "display:flex;gap:10px;justify-content:center;margin-top:20px;";
            const saveBtn = document.createElement("button");
            saveBtn.id = "quiz-area-save-selected"; saveBtn.type = "button"; saveBtn.disabled = true;
            saveBtn.textContent = "Save Selected";
            saveBtn.style.cssText = "background:#d97706;color:#fff;border:none;padding:10px 22px;border-radius:8px;font-weight:700;font-size:13px;cursor:not-allowed;opacity:0.55;";
            saveBtn.addEventListener("click", handleFallbackSaveSelected);
            actDiv.appendChild(saveBtn);
            quizArea.appendChild(actDiv);
        }

        const selAllBtn   = document.getElementById("quiz-area-select-all");
        const deselAllBtn = document.getElementById("quiz-area-deselect-all");
        if (selAllBtn)   selAllBtn.addEventListener("click",  () => setAllFallbackCBs(true));
        if (deselAllBtn) deselAllBtn.addEventListener("click", () => setAllFallbackCBs(false));

        updateFallbackCount();
    }

    function getFallbackCBs() { return Array.from(document.querySelectorAll("#quizArea .quiz-area-cb")); }

    function updateFallbackCount() {
        const cbs = getFallbackCBs();
        const n   = cbs.filter(cb => cb.checked).length;
        const el  = document.getElementById("quiz-area-count");
        if (el) el.textContent = n > 0 ? `${n} / ${cbs.length} selected` : "0 selected";
        const btn = document.getElementById("quiz-area-save-selected");
        if (btn) { btn.disabled = n === 0; btn.style.opacity = n === 0 ? "0.55" : "1"; btn.style.cursor = n === 0 ? "not-allowed" : "pointer"; }
    }

    function setAllFallbackCBs(state) {
        getFallbackCBs().forEach(cb => {
            cb.checked = state;
            const w = cb.closest(".quiz-item");
            if (w) { w.style.background = state ? "#eff6ff" : "#fff"; w.style.boxShadow = state ? "inset 3px 0 0 #2563eb" : "none"; }
        });
        updateFallbackCount();
    }

    async function handleFallbackSaveSelected() {
        const checked = getFallbackCBs().filter(cb => cb.checked);
        if (!checked.length) return;
        const indices = checked.map(cb => parseInt(cb.dataset.qIndex, 10));
        const btn = document.getElementById("quiz-area-save-selected");
        if (btn) { btn.textContent = "Saving…"; btn.disabled = true; }
        await callSaveSelected(
            indices,
            (result) => {
                if (btn) { btn.textContent = `✓ Saved ${result.total_items} items!`; btn.style.background = "#16a34a"; }
                setTimeout(() => { if (result.redirect_url) window.location.href = result.redirect_url; }, 900);
            },
            (msg) => {
                alert("Save Selected failed: " + msg);
                if (btn) { btn.textContent = "Save Selected"; btn.style.background = "#d97706"; btn.disabled = false; btn.style.opacity = "1"; btn.style.cursor = "pointer"; }
            }
        );
    }

}); // end DOMContentLoaded