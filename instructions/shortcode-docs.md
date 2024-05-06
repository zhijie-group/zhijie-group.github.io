# Short Codes

This document lists some of the usages of short codes in this website.



## `table`: Markdown table with caption

Native markdown table doesn't have caption. We wrote a simple [`table` shortcode](../layouts/shortcodes/table.html) to add caption to table.

**Example**: Simple table with caption

```html
{{< table title="Your caption here." >}}
| a    | b    |
| ---- | ---- |
| 1    | 2    |
{{</ table >}}
```

**Render**

<figure>
<div style="display: grid; place-items: center;">
<div style="display: flex; justify-content: center;">
<table>
<thead>
<tr>
<th>a</th>
<th>b</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>2</td>
</tr>
</tbody>
</table>
</div>
<figcaption style="font-size: 16px;" class="gray-text">Your caption here.</figcaption>
</div>
</figure>



Params:

- `title`: (`str`) caption text (markdown not supported)



---

#### TODO docs

- [ ] socialBadges
- [ ] image (and hopefully we can make markdown render our images like native)
