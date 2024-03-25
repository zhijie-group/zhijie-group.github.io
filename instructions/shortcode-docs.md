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



Params:

- `title`: (`str`) caption text (markdown not supported)



---

#### TODO docs

- [ ] socialBadges
- [ ] image (and hopefully we can make markdown render our images like native)
