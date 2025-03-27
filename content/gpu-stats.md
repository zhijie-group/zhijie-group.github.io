---
# title: "GPU Stats"
layout: "page"
type: "page"
url: "/gpu-stats/"
summary: "GPU statistics and usage information"
---

## GPU Statistics



{{< raw >}}
<div id="gpu-stats-container">正在加载GPU统计信息...</div>



<script>
document.addEventListener("DOMContentLoaded", function() {
    // fetch("https://raw.githubusercontent.com/zhijie-group/gpu_stat/master/gpu_status.json") 
    fetch("gpu_status.json")
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById("gpu-stats-container");
            container.innerHTML = data.map(machine => {
                const machineName = Object.keys(machine)[0];
                const stats = machine[machineName].split('\n').slice(1).map(line => {
                    let [gpu, utilization, mem, temp, uname] = line.split(/\s{2,}/);
                    gpu = gpu.replace('GPU', '').trim();
                    utilization = utilization.replace('utilization', '').trim();
                    mem = mem.replace('mem', '').trim();
                    temp = temp.replace('temp', '').trim();
                    uname = uname.replace('User', '').trim();
                    return `<tr><td>${gpu}</td><td>${utilization}</td><td>${mem}</td><td>${temp}</td><td>${uname}</td></tr>`;
                }).join('');
                return `
                    <h3>Machine ${machineName}</h3>
                    <table>
                        <thead>
                            <tr>
                                <th>GPU</th>
                                <th>Utilization</th>
                                <th>Memory</th>
                                <th>Temperature</th>
                                <th>User(s)</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${stats}
                        </tbody>
                    </table>
                `;
            }).join('');
        })
        .catch(error => {
            console.error("Error fetching GPU stats:", error);
            const container = document.getElementById("gpu-stats-container");
            container.innerHTML = "无法加载GPU统计信息。";
        });
});
</script>

<style>
    #gpu-stats-container {
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    #gpu-stats-container table {
        width: 80%; /* 设置表格宽度 */
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 18px;
        text-align: center;
    }
    #gpu-stats-container th, #gpu-stats-container td {
        padding: 6px 5px;
        border: 1px solid #ddd;
    }
    #gpu-stats-container th {
        background-color: #f2f2f2;
        text-align: center; /* 表头内容居中 */
    }
    #gpu-stats-container tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    #gpu-stats-container td:first-child {
        width: 30%; /* 设置第一列的宽度 */
    }
</style>

{{< /raw >}}