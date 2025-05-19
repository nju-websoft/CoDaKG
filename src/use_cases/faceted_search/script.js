// script.js

document.addEventListener('DOMContentLoaded', () => {
    // ... (之前的代码，包括 API_BASE_URL, DOM 元素获取, currentFilters, fetchedFacetDefinitions, currentDisplayedResults) ...

    const API_BASE_URL = 'http://127.0.0.1:5000';

    const facetsPanel = document.querySelector('.facets-panel');
    const facetListUL = facetsPanel.querySelector('.facet-list');
    const resultsArea = document.querySelector('.results-area');
    const appliedFiltersList = document.querySelector('.applied-filters-list');

    const searchInput = document.getElementById('search-input');
    const searchButton = document.getElementById('search-button');

    const detailsPanel = document.querySelector('.details-panel');
    const detailsContentDiv = detailsPanel.querySelector('.details-content');

    let currentFilters = {};
    let fetchedFacetDefinitions = {};
    let currentDisplayedResults = [];

    // *** 定义默认查询文本 ***
    const DEFAULT_SEARCH_QUERY = "North America Field Sampling Bird Genoscape";

    // *** 定义描述文本截断阈值 ***
    const DESCRIPTION_TRUNCATE_THRESHOLD = 100; // 例如，超过300个字符就截断

    // *** 定义 Dataset Details 面板中属性的显示顺序 ***
    const DATASET_DETAIL_PROPERTY_ORDER = [
        "distribution",
        "homepage", // homepage 也在列表中
        "publisher", // 假设你希望 publisher 在这里显示
        "publisher_wd", // publisher_wd 也在列表中
        "created",
        "modified",
        "themes", // themes 也在列表中
        // "creator",
        // "license",
        // 添加你希望按照特定顺序显示的任何其他属性键
        // 如果某个属性在数据中不存在，或者上面已经单独处理（Title, description），它会在这里被跳过
    ];

    // --- 异步函数：从后端获取分面信息并渲染到页面 (保持不变) ---
    async function fetchAndRenderFacets() {
       // ... (保持原来的 fetchAndRenderFacets 函数代码) ...
        try {
            const response = await fetch(`${API_BASE_URL}/facets`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const facets = await response.json();

            fetchedFacetDefinitions = facets.reduce((acc, facet) => {
                acc[facet.key] = facet;
                return acc;
            }, {});

            facetListUL.innerHTML = '';

            facets.forEach(facet => {
                const facetItemLI = document.createElement('li');
                facetItemLI.classList.add('facet-item');
                facetItemLI.dataset.facetKey = facet.key;

                const facetHeaderDiv = document.createElement('div');
                facetHeaderDiv.classList.add('facet-header');
                 facetHeaderDiv.innerHTML = `
                     <span class="facet-name">${facet.label}</span>
                     <span class="filter-icon">▼</span>
                 `;
                facetItemLI.appendChild(facetHeaderDiv);

                const facetOptionsDiv = document.createElement('div');
                facetOptionsDiv.classList.add('facet-options');

                if (facet.type === 'categorical') {
                    const optionsListUL = document.createElement('ul');
                    optionsListUL.classList.add('options-list');

                    facet.options.forEach(option => {
                        const isChecked = currentFilters[facet.key] && Array.isArray(currentFilters[facet.key]) && currentFilters[facet.key].includes(option.value);
                        const optionLI = document.createElement('li');
                        optionLI.innerHTML = `
                            <label>
                                <input type="checkbox" value="${option.value}" ${isChecked ? 'checked' : ''}>
                                ${option.value} <span>${option.count}</span>
                            </label>
                        `;
                        optionsListUL.appendChild(optionLI);
                    });
                    facetOptionsDiv.appendChild(optionsListUL);

                } else if (facet.type === 'numerical') {
                    const numericFilterDiv = document.createElement('div');
                    numericFilterDiv.classList.add('numeric-filter');
                    const currentRange = currentFilters[facet.key] || {};
                    numericFilterDiv.innerHTML = `
                        <p>Min: <input type="number" class="min-input" value="${currentRange.min !== undefined && currentRange.min !== null ? currentRange.min : ''}" step="any"></p>
                        <p>Max: <input type="number" class="max-input" value="${currentRange.max !== undefined && currentRange.max !== null ? currentRange.max : ''}" step="any"></p>
                    `;
                     facetOptionsDiv.appendChild(numericFilterDiv);

                } else if (facet.type === 'date') {
                     const dateFilterDiv = document.createElement('div');
                     dateFilterDiv.classList.add('date-filter');
                     const currentRange = currentFilters[facet.key] || {};
                     dateFilterDiv.innerHTML = `
                         <p>Start Date:</p>
                         <p><input type="text" class="start-date-input" placeholder="YYYY-MM-DD" value="${currentRange.start_date || ''}"></p>
                         <p>End Date:</p>
                         <p><input type="text" class="end-date-input" placeholder="YYYY-MM-DD" value="${currentRange.end_date || ''}"></p>
                     `;
                     facetOptionsDiv.appendChild(dateFilterDiv);
                }

                 const optionsFooterDiv = document.createElement('div');
                 optionsFooterDiv.classList.add('options-footer');
                 optionsFooterDiv.innerHTML = `
                     <button class="reset-button">Reset</button>
                     <button class="apply-button">Apply</button>
                 `;
                 facetOptionsDiv.appendChild(optionsFooterDiv);


                facetItemLI.appendChild(facetOptionsDiv);
                facetListUL.appendChild(facetItemLI);

                 if (currentFilters[facet.key] && (Array.isArray(currentFilters[facet.key]) && currentFilters[facet.key].length > 0) || (typeof currentFilters[facet.key] === 'object' && Object.keys(currentFilters[facet.key]).length > 0)) {
                     facetItemLI.classList.add('expanded');
                 }
            });

            addFacetEventListeners();
            initFlatpickr(); // 初始化日期选择器

        } catch (error) {
            console.error('Error fetching facets:', error);
            facetsPanel.innerHTML = '<p>Error loading filters.</p>';
        }
    }

     // --- 为分面项添加展开/收起、Apply/Reset按钮事件监听器 (保持不变) ---
     function addFacetEventListeners() {
        // ... (保持原来的 addFacetEventListeners 函数代码) ...
         const facetItems = facetListUL.querySelectorAll('.facet-item');

         facetItems.forEach(item => {
             const facetHeader = item.querySelector('.facet-header');
             const applyButton = item.querySelector('.apply-button');
             const resetButton = item.querySelector('.reset-button');
             const facetKey = item.dataset.facetKey;
             const facetOptionsDiv = item.querySelector('.facet-options');

             if (facetHeader) {
                  facetHeader.addEventListener('click', () => {
                     item.classList.toggle('expanded');
                 });
             }

             if (applyButton) {
                 applyButton.addEventListener('click', (event) => {
                     event.stopPropagation();

                     const facetDefinition = fetchedFacetDefinitions[facetKey];
                     if (!facetDefinition) {
                         console.error(`Facet definition not found for key: ${facetKey}`);
                         return;
                     }

                     let selectedFilterValue = null;

                     if (facetDefinition.type === 'categorical') {
                         const selectedCheckboxes = facetOptionsDiv.querySelectorAll('input[type="checkbox"]:checked');
                         selectedFilterValue = Array.from(selectedCheckboxes).map(cb => cb.value);
                         if (selectedFilterValue.length === 0) {
                             selectedFilterValue = null;
                         }

                     } else if (facetDefinition.type === 'numerical') {
                         const minInput = facetOptionsDiv.querySelector('.min-input');
                         const maxInput = facetOptionsDiv.querySelector('.max-input');
                         const minVal = minInput && minInput.value.trim() !== '' ? parseFloat(minInput.value) : null;
                         const maxVal = maxInput && maxInput.value.trim() !== '' ? parseFloat(maxInput.value) : null;

                         const isValidMin = minVal !== null && !isNaN(minVal);
                         const isValidMax = maxVal !== null && !isNaN(maxVal);

                         if (isValidMin || isValidMax) {
                             selectedFilterValue = {
                                 min: isValidMin ? minVal : null,
                                 max: isValidMax ? maxVal : null
                             };
                             if(selectedFilterValue.min === null && selectedFilterValue.max === null) {
                                  selectedFilterValue = null;
                             }
                         } else {
                             selectedFilterValue = null;
                         }

                     } else if (facetDefinition.type === 'date') {
                         const startDateInput = facetOptionsDiv.querySelector('.start-date-input');
                         const endDateInput = facetOptionsDiv.querySelector('.end-date-input');
                         const startDateVal = startDateInput && startDateInput.value.trim() !== '' ? startDateInput.value : null;
                         const endDateVal = endDateInput && endDateInput.value.trim() !== '' ? endDateInput.value : null;

                         if (startDateVal !== null || endDateVal !== null) {
                             selectedFilterValue = { start_date: startDateVal, end_date: endDateVal };
                             if(selectedFilterValue.start_date === null && selectedFilterValue.end_date === null) {
                                  selectedFilterValue = null;
                             }
                         } else {
                             selectedFilterValue = null;
                         }
                     }

                     if (selectedFilterValue !== null) {
                         currentFilters[facetKey] = selectedFilterValue;
                     } else {
                         delete currentFilters[facetKey];
                     }

                     console.log("Current Filters:", currentFilters);

                     triggerSearch();
                     renderAppliedFilters();
                 });
             }

              if (resetButton) {
                 resetButton.addEventListener('click', (event) => {
                     event.stopPropagation();

                     const facetDefinition = fetchedFacetDefinitions[facetKey];
                      if (!facetDefinition) {
                         console.error(`Facet definition not found for key: ${facetKey}`);
                         return;
                     }

                     if (facetDefinition.type === 'categorical') {
                         facetOptionsDiv.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
                     } else if (facetDefinition.type === 'numerical') {
                         facetOptionsDiv.querySelectorAll('input[type="number"]').forEach(input => input.value = '');
                     } else if (facetDefinition.type === 'date') {
                          facetOptionsDiv.querySelectorAll('input[type="text"][class$="-date-input"]').forEach(input => {
                               if (input._flatpickr) {
                                  input._flatpickr.clear();
                               } else {
                                   input.value = '';
                               }
                          });
                     }

                     delete currentFilters[facetKey];

                     console.log("Current Filters after Reset:", currentFilters);

                     triggerSearch();
                     renderAppliedFilters();
                 });
             }
         });
     }

    // --- 初始化 flatpickr 到所有的日期文本输入框 (保持不变) ---
    function initFlatpickr() {
        // ... (保持原来的 initFlatpickr 函数代码) ...
         const dateInputs = facetListUL.querySelectorAll('.facet-item input[type="text"].start-date-input, .facet-item input[type="text"].end-date-input');

         dateInputs.forEach(input => {
             if (!input._flatpickr) {
                 flatpickr(input, {
                     dateFormat: "Y-m-d",
                     allowInput: true,
                 });
             }
         });
    }


    // --- 异步函数：从后端获取搜索结果并渲染到页面 (保持不变) ---
    async function fetchAndRenderResults(filters, query) {
        resultsArea.innerHTML = 'Loading results...';

        try {
            const response = await fetch(`${API_BASE_URL}/search`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ filters: filters, query: query })
            });

            if (!response.ok) {
                 const errorText = await response.text();
                 throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            currentDisplayedResults = await response.json();

            resultsArea.innerHTML = '';

            hideDetailsPanel();

            if (currentDisplayedResults.length === 0 && (Object.keys(filters).length === 0 && query.trim() === '')) {
                 resultsArea.innerHTML = '<p>Enter a search query or apply filters to find datasets.</p>';
            } else if (currentDisplayedResults.length === 0) {
                 resultsArea.innerHTML = '<p>No results found matching your criteria.</p>';
            } else {
                 currentDisplayedResults.forEach(item => {
                       const resultCardDiv = document.createElement('div');
                       resultCardDiv.classList.add('result-card');
                       resultCardDiv.dataset.itemId = item.id;

                       // *** 卡片中只显示 Title, Contribution, Location ***
                       let cardContent = `
                           <div class="card-header">${item.title || 'N/A'}</div>
                           <div class="card-subheader">${item.publisher || 'N/A'} - ${item.publisher_wd || 'N/A'}</div>
                           <div class="card-property">
                              <span class="property-label">created:</span>
                              <span class="property-value">${item.created || ''}<br></span>
                              <span class="property-label">modified:</span>
                              <span class="property-value">${item.modified || ''}<br></span>
                              <span class="property-label">themes:</span>
                              <span class="property-value">${item.themes || ''}</span>
                           </div>
                       `;
                        // 移除遍历所有属性的循环

                       resultCardDiv.innerHTML = cardContent;
                       resultsArea.appendChild(resultCardDiv);
                   });
            }

            addResultCardEventListeners();

        } catch (error) {
            console.error('Error fetching search results:', error);
            resultsArea.innerHTML = `<p>Error loading results: ${error.message}</p>`;
             addResultCardEventListeners();
             currentDisplayedResults = [];
             hideDetailsPanel();
        }
    }

    // --- 为结果卡片添加点击高光和显示相关数据集的事件监听器 (保持不变) ---
    function addResultCardEventListeners() {
        // ... (保持原来的 addResultCardEventListeners 函数代码) ...
         const resultCards = resultsArea.querySelectorAll('.result-card');
         const mainContentDiv = document.querySelector('.main-content');

         resultCards.forEach(card => {
             card.addEventListener('click', () => {
                 const currentHighlightedCard = resultsArea.querySelector('.result-card.highlighted-card');

                 if (currentHighlightedCard === card) {
                     card.classList.remove('highlighted-card');
                     hideDetailsPanel();
                 } else {
                     if (currentHighlightedCard) {
                          currentHighlightedCard.classList.remove('highlighted-card');
                     }

                     card.classList.add('highlighted-card');
                     showDetailsPanel(card.dataset.itemId);
                 }
             });
         });
    }

     // --- 渲染选定数据集详细信息和相关数据集到右侧面板 (修改，显示 graph 属性) ---
     function showDetailsPanel(itemId) {

        const selectedItemId = itemId;
        const selectedItem = currentDisplayedResults.find(item => {
            return item && item.id !== undefined && item.id === selectedItemId;
        });

        const mainContentDiv = document.querySelector('.main-content');
        const detailsContentDiv = document.querySelector('.details-content');

        if (selectedItem) {
            let detailsHtml = `<h4>${selectedItem.Title || selectedItem.title || 'N/A'}</h4>`; // 使用 Title 或 title

            // 处理 description 展开/折叠 (保持不变)
            const descriptionText = selectedItem.description || '';
            if (descriptionText.length > DESCRIPTION_TRUNCATE_THRESHOLD) {
                const truncatedText = descriptionText.substring(0, DESCRIPTION_TRUNCATE_THRESHOLD) + '...';
                detailsHtml += `
                    <p class="detail-description">
                        <strong>Description:</strong>
                        <span class="description-truncated">${truncatedText}</span>
                        <span class="description-full" style="display: none;">${descriptionText}</span>
                        <a href="#" class="read-more-toggle" data-target="description">Read More</a>
                    </p>
                `;
            } else if (descriptionText) {
                 detailsHtml += `
                     <p class="detail-description">
                         <strong>Description:</strong>
                         <span>${descriptionText}</span>
                     </p>
                 `;
            }


            // 按照指定顺序遍历并显示属性 (保持不变)
            DATASET_DETAIL_PROPERTY_ORDER.forEach(key => {
                // ... (属性遍历显示逻辑保持不变) ...
                 if (selectedItem.hasOwnProperty(key) && key !== 'id' && key !== 'Title' && key !== 'title' && key !== 'description' && key !== 'related_datasets' && key !== 'graph') { // *** 排除 graph 属性 ***
                     const value = selectedItem[key];
                     const propertyLabel = fetchedFacetDefinitions[key] ? fetchedFacetDefinitions[key].label : key;
                     const propertyValue = value !== null && value !== undefined && value !== '' ? value : '';

                     // 特殊处理 publisher_wd 为 Wikidata 链接
                     if (key === 'publisher_wd' && propertyValue && propertyValue !== 'Empty') {
                         detailsHtml += `
                             <div class="detail-property">
                                <strong>publisher wikidata:</strong>
                                <a href="https://www.wikidata.org/wiki/${propertyValue}" target="_blank">${propertyValue}</a>
                             </div>
                         `;
                     // 特殊处理 homepage 为超链接
                     } else if (key === 'homepage' && propertyValue && propertyValue !== 'Empty') {
                          detailsHtml += `
                             <div class="detail-property detail-homepage">
                                <strong>homepage:</strong>
                                <a href="${propertyValue}" target="_blank">${propertyValue}</a>
                             </div>
                         `;
                     // 处理 themes (显示为列表)
                     } else if (key === 'themes' && Array.isArray(propertyValue)) {
                          if (propertyValue.length > 0) {
                               detailsHtml += `<div class="detail-property"><strong>Themes:</strong><ul>`;
                               propertyValue.forEach(theme => {
                                   detailsHtml += `<li>${theme}</li>`;
                               });
                               detailsHtml += `</ul></div>`;
                          } else {
                               detailsHtml += `<div class="detail-property"><strong>Themes:</strong> Empty</div>`;
                          }
                     // 显示其他普通属性 (如果值不为空)
                     } else if (propertyValue !== 'Empty') {
                          detailsHtml += `
                              <div class="detail-property detail-property-${key.toLowerCase()}">
                                 <strong>${propertyLabel}:</strong>
                                 <span>${propertyValue}</span>
                              </div>
                          `;
                     }
                 }
            });

            console.log(selectedItem);
             // *** 在其他属性之后，相关数据集列表之前，显示 graph 属性对应的图片 ***
             if (selectedItem.graph) {
                console.log('graph');
                 const graphPath = selectedItem.graph;
                 // *** 注意：如果路径是 .pdf 文件，<img> 标签无法直接显示 PDF 内容 ***
                 // 如果确定路径是图片文件 (png, jpg, gif 等)，可以使用 <img>
                 // 如果必须显示 PDF，需要使用 <object> 或 <iframe>，或者一个 PDF 预览库
                 // 这里假设路径是图片文件，使用 <img>
                 detailsHtml += `
                     <div class="detail-graph">
                         <strong>Graph:</strong>
                         <img src="${graphPath}" alt="Dataset Graph" class="dataset-graph-image">
                     </div>
                 `;
                  // 如果路径是 PDF，可以考虑使用 object 或 iframe:
                  /*
                  detailsHtml += `
                       <div class="detail-graph">
                            <strong>Graph:</strong>
                            <object data="${graphPath}" type="application/pdf" width="100%" height="500px">
                                 <p>It appears you don't have a PDF plugin for this browser.
                                 No biggie... you can <a href="${graphPath}">click here to download the PDF file.</a></p>
                            </object>
                       </div>
                  `;
                  */
             }


            // 显示相关数据集列表 (先显示 Title，再循环其他属性) (保持不变)
            if (selectedItem.related_datasets && selectedItem.related_datasets.length > 0) {
                 // ... (相关数据集列表 HTML 生成代码保持不变) ...

                detailsHtml += '<h4>Related Datasets:</h4><ul class="related-list">';
                selectedItem.related_datasets.forEach(relatedItem => {
                    detailsHtml += `<li>`;

                    const relatedTitle = relatedItem.title || relatedItem.Title || 'N/A';
                    detailsHtml += `<p class="related-item-title">${relatedTitle}</p>`;

                    for (const key in relatedItem) {

                         if (relatedItem.hasOwnProperty(key) && key !== 'id' && key !== 'Title' && key !== 'title') {
                             const value = relatedItem[key];
                             const facetDefinition = fetchedFacetDefinitions[key];
                             const propertyLabel = facetDefinition ? facetDefinition.label : key;
                             const propertyValue = value !== null && value !== undefined && value !== '' ? value : 'Empty';

                             if (key === 'publisher_wd' && propertyValue && propertyValue !== 'Empty') {
                                 detailsHtml += `<p class="related-item-${key.toLowerCase()}">publisher wikidata: ${propertyValue}</p>`;
                             } else if (key === 'homepage' && propertyValue && propertyValue !== 'Empty') {
                                  detailsHtml += `
                                     <p class="related-item-${key.toLowerCase()}">
                                        <strong>homepage:</strong>
                                        <a href="${propertyValue}" target="_blank">${propertyValue}</a>
                                     </p>
                                 `;
                             } else if (key === 'relationships' && propertyValue && propertyValue !== 'Empty') {
                                  detailsHtml += `<p class="related-item-relationships">relationships: ${propertyValue}</p>`;
                             } else if (propertyValue !== 'Empty') {
                                 detailsHtml += `<p class="related-item-${key.toLowerCase()}">${propertyLabel}: ${propertyValue}</p>`;
                             }
                         }
                     }

                     detailsHtml += `</li>`;
                 });
                 detailsHtml += '</ul>';
             } else {
                 detailsHtml += '<p>No related datasets found.</p>';
             }


            detailsContentDiv.innerHTML = detailsHtml;

            addReadMoreToggleListeners(); // 添加展开/折叠事件监听器

            const mainContentDiv = document.querySelector('.main-content');
            detailsPanel.style.display = 'flex';
            mainContentDiv.classList.add('details-visible');

        } else {
             console.warn("Dataset item with ID", itemId, "not found in currentDisplayedResults.");
             hideDetailsPanel();
             detailsContentDiv.innerHTML = '<p>Details not available for this item.</p>';
         }
   }

     // *** 新增函数：添加 Read More/Show Less 事件监听器 ***
     function addReadMoreToggleListeners() {
        // 查找详情面板中所有的展开/折叠切换元素
        const toggles = detailsContentDiv.querySelectorAll('.read-more-toggle');

        toggles.forEach(toggle => {
            toggle.addEventListener('click', (event) => {
                event.preventDefault(); // 阻止默认链接行为

                const targetType = toggle.dataset.target; // 获取目标类型，例如 'description'

                if (targetType === 'description') {
                    const descriptionPara = toggle.closest('.detail-description'); // 找到最近的 description 段落

                    if (descriptionPara) {
                        const truncatedSpan = descriptionPara.querySelector('.description-truncated');
                        const fullSpan = descriptionPara.querySelector('.description-full');

                        if (truncatedSpan && fullSpan) {
                            if (truncatedSpan.style.display !== 'none') {
                                // 当前显示的是截断文本，点击后显示完整文本
                                truncatedSpan.style.display = 'none';
                                fullSpan.style.display = 'inline'; // 使用 inline 或 block，取决于你希望完整文本如何显示
                                toggle.textContent = 'Show Less'; // 改变链接文本
                            } else {
                                // 当前显示的是完整文本，点击后显示截断文本
                                fullSpan.style.display = 'none';
                                truncatedSpan.style.display = 'inline';
                                toggle.textContent = 'Read More'; // 改变链接文本
                            }
                        }
                    }
                }
                // 如果将来有其他需要展开/折叠的元素，可以在这里添加 else if (targetType === '...')
            });
        });
    }

     // --- 隐藏右侧详情面板 (保持不变) ---
     function hideDetailsPanel() {
         // ... (保持原来的 hideDetailsPanel 函数代码) ...
         const mainContentDiv = document.querySelector('.main-content');
         detailsPanel.style.display = 'none';
         mainContentDiv.classList.remove('details-visible');
         detailsContentDiv.innerHTML = '<p>Click on a dataset card to see details.</p>';
         // 确保取消高亮选中的卡片
         const currentHighlightedCard = resultsArea.querySelector('.result-card.highlighted-card');
         if (currentHighlightedCard) {
              currentHighlightedCard.classList.remove('highlighted-card');
         }
     }


     // --- 渲染已应用的过滤器标签 (保持不变) ---
     function renderAppliedFilters() {
        // ... (保持原来的 renderAppliedFilters 函数代码) ...
         appliedFiltersList.innerHTML = '';

         for (const facetKey in currentFilters) {
             if (currentFilters.hasOwnProperty(facetKey)) {
                 const filterValue = currentFilters[facetKey];
                 const facetDefinition = fetchedFacetDefinitions[facetKey];
                 if (!facetDefinition) {
                     console.warn(`Facet definition not found for key: ${facetKey}, skipping applied filter tag.`);
                     continue;
                 }

                 const facetLabel = facetDefinition.label;
                 const facetType = facetDefinition.type;

                 let tagText = `${facetLabel}: `;

                 if (facetType === 'categorical') {
                     if (Array.isArray(filterValue)) {
                          tagText += filterValue.join(', ');
                     } else {
                         tagText += filterValue;
                     }

                 } else if (facetType === 'numerical') {
                      if (typeof filterValue === 'object' && filterValue !== null) {
                          const min = filterValue.min;
                          const max = filterValue.max;
                          if (min !== null && max !== null) {
                              tagText += `between ${min} and ${max}`;
                          } else if (min !== null) {
                              tagText += `after ${min}`;
                          } else if (max !== null) {
                              tagText += `before ${max}`;
                          } else { continue; }
                     } else { continue; }

                 } else if (facetType === 'date') {
                      if (typeof filterValue === 'object' && filterValue !== null) {
                          const startDate = filterValue.start_date;
                          const endDate = filterValue.end_date;
                           if (startDate && endDate) {
                               tagText += `${startDate} to ${endDate}`;
                           } else if (startDate) {
                                tagText += `after ${startDate}`;
                           } else if (endDate) {
                                tagText += `before ${endDate}`;
                           } else { continue; }
                     } else { continue; }

                 } else {
                     tagText += filterValue;
                 }

                 if (`${facetLabel}: ` === tagText && facetType !== 'categorical') {
                      continue;
                 }

                 const filterTagSpan = document.createElement('span');
                 filterTagSpan.classList.add('filter-tag');
                 filterTagSpan.dataset.facetKey = facetKey;
                 filterTagSpan.innerHTML = `
                     ${tagText}
                     <span class="close-tag">&times;</span>
                 `;
                 appliedFiltersList.appendChild(filterTagSpan);
             }
         }
     }


     // --- 为已应用过滤器标签的关闭按钮添加事件监听器 (保持不变) ---
      if (appliedFiltersList) {
         appliedFiltersList.addEventListener('click', (event) => {
            if (event.target.classList.contains('close-tag')) {
                const filterTag = event.target.closest('.filter-tag');
                if (filterTag) {
                    const facetKeyToRemove = filterTag.dataset.facetKey;

                    delete currentFilters[facetKeyToRemove];

                    console.log("Current Filters after removing tag:", currentFilters);

                    triggerSearch();
                    renderAppliedFilters();

                    // 移除过滤条件后，隐藏详情面板并取消高亮选中的卡片
                    hideDetailsPanel();

                    const correspondingFacetItem = facetListUL.querySelector(`.facet-item[data-facet-key="${facetKeyToRemove}"]`);
                    if (correspondingFacetItem) {
                        const facetDefinition = fetchedFacetDefinitions[facetKeyToRemove];
                        if (facetDefinition) {
                             const facetOptionsDiv = correspondingFacetItem.querySelector('.facet-options');
                             if (facetDefinition.type === 'categorical') {
                                 facetOptionsDiv.querySelectorAll('input[type="checkbox"]').forEach(cb => cb.checked = false);
                             } else if (facetDefinition.type === 'numerical') {
                                 facetOptionsDiv.querySelectorAll('input[type="number"]').forEach(input => input.value = '');
                             } else if (facetDefinition.type === 'date') {
                                  facetOptionsDiv.querySelectorAll('input[type="text"][class$="-date-input"]').forEach(input => {
                                      if (input._flatpickr) {
                                          input._flatpickr.clear();
                                      } else {
                                          input.value = '';
                                      }
                                  });
                             }
                        }
                    }
                }
            }
        });
     }

     // --- 触发搜索的函数 (保持不变) ---
     function triggerSearch() {
         const query = searchInput.value;
         // 当触发搜索时，移除默认文本类，因为这是用户明确的操作
         searchInput.classList.remove('default-text');
         fetchAndRenderResults(currentFilters, query);
     }

     // --- 为搜索按钮和输入框添加事件监听器 (修改，添加焦点/失焦/输入事件) ---
     if (searchButton) {
         searchButton.addEventListener('click', triggerSearch);
     }

     if (searchInput) {
         // 监听回车键
         searchInput.addEventListener('keypress', (event) => {
             if (event.key === 'Enter') {
                 event.preventDefault();
                 triggerSearch();
             }
         });

         // *** 添加焦点事件：如果内容是默认文本，则清空输入框并移除默认类 ***
         searchInput.addEventListener('focus', () => {
             if (searchInput.value === DEFAULT_SEARCH_QUERY && searchInput.classList.contains('default-text')) {
                 searchInput.value = '';
                 searchInput.classList.remove('default-text');
             }
         });

         // *** 添加失焦事件：如果输入框变为空，则恢复默认文本并添加默认类 ***
         searchInput.addEventListener('blur', () => {
             if (searchInput.value.trim() === '') {
                 searchInput.value = DEFAULT_SEARCH_QUERY;
                 searchInput.classList.add('default-text');
             }
         });

         // *** 添加输入事件：一旦用户输入内容，移除默认类 ***
         searchInput.addEventListener('input', () => {
              // 当用户输入时，即使他们重新输入了默认文本，也保持黑色字体
              if (searchInput.classList.contains('default-text') && searchInput.value !== DEFAULT_SEARCH_QUERY) {
                   searchInput.classList.remove('default-text');
              }
              // 如果用户删除了所有内容，blur 事件会处理恢复默认文本
         });
     }


    // --- 页面加载完成后 ---
    fetchAndRenderFacets().then(() => {
        // 设置默认查询文本和初始类
        searchInput.value = DEFAULT_SEARCH_QUERY;
        searchInput.classList.add('default-text'); // *** 初始添加默认类 ***

        // 页面初始加载时，显示默认查询文本，但不立即执行搜索，结果区域初始为空
        // triggerSearch(); // 如果需要初始加载时显示默认搜索结果，取消注释此行

         renderAppliedFilters();

         resultsArea.innerHTML = '<p>Enter a search query or apply filters to find datasets.</p>';
         addResultCardEventListeners();
         hideDetailsPanel();
    });
});