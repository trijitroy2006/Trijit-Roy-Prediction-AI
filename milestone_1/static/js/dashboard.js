document.addEventListener('DOMContentLoaded', () => {
    const form = document.querySelector('.submission-form');
    if (!form) return;

    // Elements to update
    const updateMetrics = (data) => {
        const metricValues = document.querySelectorAll('.metric-value');
        if (metricValues.length === 3 && data.market_data) {
            metricValues[0].textContent = data.market_data.TAM.value;
            metricValues[1].textContent = data.market_data.SAM.value;
            metricValues[2].textContent = data.market_data.SOM.value;
        }

        // Competitor List
        const compList = document.querySelector('.competitor-list');
        if (compList && data.competitors) {
            compList.innerHTML = '';
            data.competitors.forEach(comp => {
                const compTypeClass = comp.type.toLowerCase();
                const card = `
                    <div class="competitor-card">
                        <div class="comp-header">
                            <h3>${comp.name}</h3>
                            <span class="tag ${compTypeClass}">${comp.type}</span>
                        </div>
                        <div class="comp-metrics">
                            <div class="c-metric">
                                <span class="c-label">Market Share</span>
                                <span class="c-val">${comp.market_share}</span>
                            </div>
                            <div class="c-metric">
                                <span class="c-label">Revenue</span>
                                <span class="c-val">${comp.revenue}</span>
                            </div>
                            <div class="c-metric">
                                <span class="c-label">Growth</span>
                                <span class="c-val positive">${comp.growth}</span>
                            </div>
                        </div>
                        <div class="comp-progress">
                            <span class="c-label">Market Position</span>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${comp.position}%"></div>
                            </div>
                        </div>
                    </div>
                `;
                compList.insertAdjacentHTML('beforeend', card);
            });
        }
    };

    const fetchAnalytics = async () => {
        const formData = new FormData(form);
        const dataObj = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(dataObj)
            });

            if (response.ok) {
                const result = await response.json();
                updateMetrics(result);
            }
        } catch (err) {
            console.error("Failed to fetch dynamic data:", err);
        }
    };

    // Attach listeners to all inputs and selects
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('input', () => {
            // Only update if we already have the competitor-list showing
            const compList = document.querySelector('.competitor-list');
            if (compList) {
                fetchAnalytics();
            }
        });
    });
});
