// static/js/create_tos.js
document.addEventListener("DOMContentLoaded", () => {

    // ── DOM refs ──
    const addRowBtn           = document.getElementById("addRow");
    const generateBtn         = document.getElementById("generateTOSQuizBtn");
    const tosTable            = document.getElementById("tosTable");
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

    let tests            = [];
    let abortController  = null;
    let currentMasterId  = null;
    let progressInterval = null;

    if (percentValidationMsg) percentValidationMsg.style.display = "none";

    // ================================================================
    // FILE PREVIEW SYSTEM
    // ================================================================
    // Each topic row gets a hidden <input type="file"> + custom UI.
    // When user picks a file, we display name/size + Open/Clear buttons.
    // "Open" creates a blob URL and opens in new tab — works for PDFs
    // and images natively; DOCX/PPTX get downloaded so user can verify
    // in their default app.

    /** File-type icon by extension */
    function fileIcon(filename) {
        const ext = (filename.split('.').pop() || '').toLowerCase();
        const map = {
            pdf:  '📕',
            doc:  '📘', docx: '📘',
            ppt:  '📙', pptx: '📙',
            txt:  '📄', md: '📄',
            png:  '🖼️', jpg: '🖼️', jpeg: '🖼️', gif: '🖼️', webp: '🖼️',
        };
        return map[ext] || '📎';
    }

    /** Pretty-format file size */
    function formatSize(bytes) {
        if (bytes < 1024) return bytes + ' B';
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
        return (bytes / 1024 / 1024).toFixed(2) + ' MB';
    }

    /** Per-row blob URL — keep reference so we can revoke on clear/replace */
    function clearBlobUrl(row) {
        const url = row.dataset.previewUrl;
        if (url) {
            URL.revokeObjectURL(url);
            delete row.dataset.previewUrl;
        }
    }

    /** Wire up a single file-input cell. Idempotent. */
    function wireFileCell(row) {
        const input    = row.querySelector('.learnFile');
        const pickBtn  = row.querySelector('.file-pick-btn');
        const meta     = row.querySelector('.file-meta');
        const nameEl   = row.querySelector('.file-meta-name');
        const sizeEl   = row.querySelector('.file-meta-size');
        const iconEl   = row.querySelector('.file-meta-icon');
        const openBtn  = row.querySelector('.file-preview-btn');
        const clearBtn = row.querySelector('.file-clear-btn');
if (openBtn) openBtn.style.display = 'none'; // hidden until a previewable file is chosen
        if (!input || !pickBtn) return;
        if (input.dataset.wired === '1') return;
        input.dataset.wired = '1';

        // Click on "Choose file…" → open native picker
        pickBtn.addEventListener('click', () => input.click());

        // File chosen
        input.addEventListener('change', () => {
            const file = input.files && input.files[0];
            clearBlobUrl(row);

            if (!file) {
                pickBtn.classList.remove('has-file');
                pickBtn.innerHTML = '📎 Choose file…';
                if (meta) meta.classList.remove('show');
                const previewExts = ['pdf', 'png', 'jpg', 'jpeg', 'gif', 'webp', 'txt', 'md'];
const fileExt = (file.name.split('.').pop() || '').toLowerCase();
if (openBtn) openBtn.style.display = previewExts.includes(fileExt) ? '' : 'none';
                return;
            }

            pickBtn.classList.add('has-file');
            pickBtn.innerHTML = '✓ Replace file…';
            if (iconEl) iconEl.textContent = fileIcon(file.name);
            if (nameEl) {
                nameEl.textContent = file.name;
                nameEl.title = file.name;
            }
            if (sizeEl) sizeEl.textContent = formatSize(file.size);
            if (meta)   meta.classList.add('show');

            // Pre-create the blob URL — preview button is now ready instantly
            row.dataset.previewUrl = URL.createObjectURL(file);
        });

        // "👁 Open" — open blob URL in new tab
       // "👁 Open" — show inline preview modal
if (openBtn) {
    openBtn.addEventListener('click', () => {
        const file = input.files && input.files[0];
        if (!file) return;
        const url = row.dataset.previewUrl || URL.createObjectURL(file);
        row.dataset.previewUrl = url;

        const modal    = document.getElementById('filePreviewModal');
        const body     = document.getElementById('filePreviewBody');
        const title    = document.getElementById('filePreviewTitle');
        const fallback = document.getElementById('filePreviewFallback');
        const dlLink   = document.getElementById('filePreviewDownload');
        if (!modal || !body) return;

        title.textContent = file.name;
        body.innerHTML    = '';
        fallback.style.display = 'none';

        const ext = (file.name.split('.').pop() || '').toLowerCase();
        const previewable = ['pdf', 'png', 'jpg', 'jpeg', 'gif', 'webp', 'txt', 'md'];

        if (previewable.includes(ext)) {
            if (['png','jpg','jpeg','gif','webp'].includes(ext)) {
                // Image: show <img>
                const img = document.createElement('img');
                img.src = url;
                img.style.cssText = 'max-width:100%; max-height:100%; object-fit:contain; padding:16px;';
                body.appendChild(img);
            } else {
                // PDF / txt / md: show in iframe
                const iframe = document.createElement('iframe');
                iframe.src = url;
                iframe.style.cssText = 'width:100%; height:100%; border:none; flex:1; display:block;';
                body.style.display = 'block';
                body.appendChild(iframe);
            }
        } else {
            // DOCX / PPTX etc. — show download fallback
            body.style.display = 'none';
            fallback.style.display = 'flex';
            if (dlLink) { dlLink.href = url; dlLink.download = file.name; }
        }

        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
    });
}

        // "✕" — clear selection
        if (clearBtn) {
            clearBtn.addEventListener('click', () => {
                clearBlobUrl(row);
                input.value = ''; // reset so re-picking same file fires change
                pickBtn.classList.remove('has-file');
                pickBtn.innerHTML = '📎 Choose file…';
                if (meta) meta.classList.remove('show');
                if (openBtn) openBtn.style.display = 'none'; // ← ADD THIS
            });
        }
    }

    /** Wire all currently-rendered file cells */
    function wireAllFileCells() {
        tosTable.querySelectorAll('tbody tr').forEach(wireFileCell);
    }

    wireAllFileCells();

    // ================================================================
    // PROGRESS INDICATOR
    // ================================================================
    let _progressTotal = 10;

    function startProgress(totalItems) {
        _progressTotal = Math.max(1, totalItems || 10);
        if (spinnerCount)    spinnerCount.textContent  = '0';
        if (loadingStatus)   loadingStatus.textContent = 'Analyzing materials…';
        if (progressBarFill) progressBarFill.style.width = '0%';
        setStep('analyze');
        let ap = 0;
        const analyzeTimer = setInterval(() => {
            ap = Math.min(ap + 4, 100);
            setBar(Math.round(ap * 0.15));
            if (ap >= 100) { clearInterval(analyzeTimer); setStep('generate'); _startPolling(); }
        }, 40);
    }

    function _startPolling() {
        progressInterval = setInterval(async () => {
            try {
                const resp = await fetch('/dashboard/generation_progress', { cache: 'no-store' });
                if (!resp.ok) return;
                const data = await resp.json();
                const current = data.current || 0;
                const total   = data.total   || _progressTotal;
                if (spinnerCount)  spinnerCount.textContent  = current;
                if (loadingStatus) loadingStatus.textContent = `Generating item ${current} of ${total}…`;
                const pct = 15 + Math.round((current / Math.max(total, 1)) * 75);
                setBar(Math.min(pct, 90));
            } catch (_) {}
        }, 800);
    }

    function stopProgress() {
        if (progressInterval) { clearInterval(progressInterval); progressInterval = null; }
        if (spinnerCount)    spinnerCount.textContent  = '✓';
        if (loadingStatus)   loadingStatus.textContent = 'Done!';
        if (progressBarFill) progressBarFill.style.width = '100%';
        setStep('finalize', true);
    }

    function resetProgress() {
        if (progressInterval) { clearInterval(progressInterval); progressInterval = null; }
        if (spinnerCount)    spinnerCount.textContent  = '0';
        if (loadingStatus)   loadingStatus.textContent = 'Starting up…';
        if (progressBarFill) progressBarFill.style.width = '0%';
        setStep('analyze');
    }

    function setBar(pct) { if (progressBarFill) progressBarFill.style.width = pct + '%'; }

    function setStep(active, allDone) {
        const steps = { analyze: stepAnalyze, generate: stepGenerate, finalize: stepFinalize };
        const order = ['analyze', 'generate', 'finalize'];
        const activeIdx = order.indexOf(active);
        order.forEach((key, i) => {
            const el = steps[key]; if (!el) return;
            el.classList.remove('active', 'done');
            if (allDone) { el.classList.add('done'); }
            else if (i < activeIdx) { el.classList.add('done'); }
            else if (i === activeIdx) { el.classList.add('active'); }
        });
    }

    // ================================================================
    // PREVIEW HELPERS
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
        setTimeout(() => { const s = previewArea.querySelector('.bondpaper'); if (s) s.scrollTop = 0; }, 40);
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
        if (countEl) countEl.textContent = checked.length > 0 ? `${checked.length} / ${all.length} selected` : '0 selected';
        if (saveBtn) saveBtn.disabled = checked.length === 0;
    }

    function setAllFragmentCBs(state) {
        getFragmentCBs().forEach(cb => { cb.checked = state; highlightFragmentItem(cb); });
        updateFragmentCount();
    }
    // ── Inline file preview modal close ──
const filePreviewModal = document.getElementById('filePreviewModal');
const filePreviewClose = document.getElementById('filePreviewClose');
if (filePreviewClose && filePreviewModal) {
    filePreviewClose.addEventListener('click', () => {
        filePreviewModal.style.display = 'none';
        document.body.style.overflow   = '';
        // clear iframe src to stop any embedded content
        const body = document.getElementById('filePreviewBody');
        if (body) body.innerHTML = '';
    });
    // also close on backdrop click
    filePreviewModal.addEventListener('click', (e) => {
        if (e.target === filePreviewModal) filePreviewClose.click();
    });
}

    // ================================================================
    // STATIC BUTTON LISTENERS
    // ================================================================
    if (staticCloseX)   staticCloseX.addEventListener('click', hidePreview);
    if (staticCloseBtn) staticCloseBtn.addEventListener('click', hidePreview);
    if (staticSaveBtn) {
        staticSaveBtn.addEventListener('click', () => {
            const r = document.querySelector('input[name="redirect_after_save"]');
            if (r && r.value) window.location = r.value; else hidePreview();
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
                if (r && r.value) window.location = r.value; else hidePreview();
                return;
            }
            if (id === 'preview-select-all-btn')  { setAllFragmentCBs(true);  return; }
            if (id === 'preview-deselect-all-btn') { setAllFragmentCBs(false); return; }

            if (id === 'quiz-preview-save-selected') {
                ev.preventDefault(); ev.stopPropagation();
                const checked = getFragmentCBs().filter(cb => cb.checked);
                if (!checked.length) return;
                const indices = checked.map(cb => parseInt(cb.dataset.qIndex, 10));
                const btn = ev.target;
                btn.textContent = 'Saving…'; btn.disabled = true; btn.style.background = '#d97706';
                callSaveSelected(indices,
                    (result) => {
                        btn.textContent = `✓ Saved ${result.total_items} items!`; btn.style.background = '#16a34a';
                        setTimeout(() => {
                            const r = document.querySelector('input[name="redirect_after_save"]');
                            if (r && r.value) window.location.href = r.value; else hidePreview();
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
                highlightFragmentItem(ev.target); updateFragmentCount();
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
            console.error("save_selected error:", err); onError("Network error. Please try again.");
        }
    }

    // ================================================================
    // HELPERS
    // ================================================================
    function _injectMasterIdInput(masterId) {
        if (!masterId) return;
        let el = document.getElementById('_master_id_input');
        if (!el) { el = document.createElement('input'); el.type = 'hidden'; el.id = '_master_id_input'; el.name = 'master_id'; document.body.appendChild(el); }
        el.value = masterId;
    }

    function _setRedirectInput(url) {
        let el = document.querySelector('input[name="redirect_after_save"]');
        if (!el) { el = document.createElement('input'); el.type = 'hidden'; el.name = 'redirect_after_save'; document.body.appendChild(el); }
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
    // TOPIC TABLE — add row now includes the file-cell markup
    // ================================================================
    addRowBtn.onclick = () => {
        const tbody = tosTable.querySelector("tbody");
        const row   = document.createElement("tr");
        row.innerHTML = `
            <td><input type="text" placeholder="e.g. Deep Learning" class="topicName"></td>
            <td><input type="number" value="3" min="1" class="topicHours" style="width:80px;"></td>
            <td class="file-cell">
                <div class="file-input-wrap">
                    <input type="file" class="learnFile" accept=".pdf,.docx,.pptx,.txt,.md">
                    <button type="button" class="file-pick-btn">📎 Choose file…</button>
                    <div class="file-meta">
                        <span class="file-meta-icon">📄</span>
                        <span class="file-meta-name">—</span>
                        <span class="file-meta-size">—</span>
                        <div class="file-actions">
                            <button type="button" class="file-action-btn file-preview-btn" title="Preview file">👁 Open</button>
                            <button type="button" class="file-action-btn danger file-clear-btn" title="Clear selection">✕</button>
                        </div>
                    </div>
                </div>
            </td>
            <td><button type="button" class="btn secondary remove">Remove</button></td>
        `;
        tbody.appendChild(row);
        wireFileCell(row);
    };

    tosTable.addEventListener("click", (e) => {
        if (e.target.classList.contains("remove")) {
            const row = e.target.closest("tr");
            clearBlobUrl(row);  // free memory
            row.remove();
        }
    });

    const toBase64 = (file) => new Promise((res, rej) => {
        const r = new FileReader(); r.readAsDataURL(file);
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
                <input type="text" class="testDesc desc-input" placeholder="Short description / instruction">
                <button class="removeTest btn secondary" type="button">✕</button>
            `;
            if (testList) testList.appendChild(div);
            updateTestCount();
        };
    }

    document.addEventListener("click", (e) => { if (e.target.classList.contains("removeTest")) { e.target.closest(".test-row").remove(); updateTestCount(); } });
    document.addEventListener("input", (e) => { if (e.target.classList.contains("testItems")) updateTestCount(); });

    if (cancelTestModalBtn) cancelTestModalBtn.onclick = () => { if (testModal) testModal.style.display = "none"; };

    // ================================================================
    // CANCEL GENERATION
    // ================================================================
    if (cancelGenerationBtn) {
        cancelGenerationBtn.onclick = () => {
            const confirmed = confirm(
                "Cancel generation?\n\nAny questions generated so far will be lost."
            );
            if (!confirmed) return;

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
            const totalQuizItems = parseInt(document.getElementById("totalQuizItemsInput").value, 10);
            resetProgress();
            if (loadingOverlay) loadingOverlay.style.display = 'flex';
            startProgress(totalQuizItems);
            submitTOSWithTests(tests);
        };
    }

    // ================================================================
    // SUBMIT  — includes CILOs in payload
    // ================================================================
    async function submitTOSWithTests(tests) {
        abortController = new AbortController();
        const signal    = abortController.signal;

        const title       = document.getElementById("tosTitle").value.trim();
        const subjectType = subjectTypeSelect ? subjectTypeSelect.value : 'nonlab';
        const totalQuiz   = parseInt(document.getElementById("totalQuizItemsInput").value, 10);

        const cilos = (typeof window.getCilos === 'function') ? window.getCilos() : [];

        const topics = [];
        for (const row of Array.from(tosTable.querySelectorAll("tbody tr"))) {
            const topicName  = row.querySelector('.topicName')  ? row.querySelector('.topicName').value.trim() : '';
            const hoursValue = row.querySelector('.topicHours') ? parseInt(row.querySelector('.topicHours').value, 10) : 0;
            const fileInput  = row.querySelector('.learnFile');
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

        const payload = { title, subjectType, totalQuizItems: totalQuiz, topics, tests, cilos };
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

        } catch (err) {
            resetProgress();
            if (loadingOverlay) loadingOverlay.style.display = 'none';
            if (err.name === 'AbortError') console.log("Fetch aborted.");
            else { console.error(err); alert("An error occurred. See console."); }
        } finally {
            abortController = null;
        }
    }

}); // end DOMContentLoaded