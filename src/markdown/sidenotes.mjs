import { defineHastPlugin } from 'satteri';

const isElement = (node, tagName) => node?.type === 'element' && node.tagName === tagName;

const isBacklink = (node) =>
	isElement(node, 'a') && Object.hasOwn(node.properties ?? {}, 'dataFootnoteBackref');

const cleanNode = (node) => {
	if (isBacklink(node)) return null;
	if (node.type === 'text') return { type: 'text', value: node.value };
	if (node.type !== 'element') return null;

	return {
		type: 'element',
		tagName: node.tagName,
		properties: { ...node.properties },
		children: node.children.map(cleanNode).filter(Boolean),
	};
};

const collectNoteContent = (listItem) => {
	const content = [];

	for (const child of listItem.children) {
		if (child.type === 'text' && child.value.trim() === '') continue;

		if (isElement(child, 'p')) {
			const paragraph = child.children.map(cleanNode).filter(Boolean);
			if (content.length && paragraph.length) content.push({ type: 'element', tagName: 'br', properties: {}, children: [] });
			content.push(...paragraph);
			continue;
		}

		const cleaned = cleanNode(child);
		if (cleaned) content.push(cleaned);
	}

	return content;
};

export const collectFootnotes = defineHastPlugin({
	name: 'phyai-collect-footnotes',
	element: {
		filter: ['section'],
		visit(node, context) {
			if (!node.properties?.dataFootnotes) return;

			const definitions = new Map();
			const list = node.children.find((child) => isElement(child, 'ol'));
			if (list) {
				for (const child of list.children) {
					if (!isElement(child, 'li') || typeof child.properties?.id !== 'string') continue;
					definitions.set(`#${child.properties.id}`, collectNoteContent(child));
				}
			}

			context.data.phyaiFootnotes = definitions;
			context.removeNode(node);
		},
	},
});

export const renderSidenotes = defineHastPlugin({
	name: 'phyai-render-sidenotes',
	element: {
		filter: ['sup'],
		visit(node, context) {
			const reference = node.children.find(
				(child) => isElement(child, 'a') && child.properties?.dataFootnoteRef,
			);
			const definitions = context.data.phyaiFootnotes;
			if (!reference || !(definitions instanceof Map)) return;

			const href = reference.properties?.href;
			const noteContent = typeof href === 'string' ? definitions.get(href) : undefined;
			if (!noteContent) return;

			const number = context.textContent(reference);
			const referenceId =
				typeof reference.properties?.id === 'string'
					? reference.properties.id
					: `footnote-${number}`;
			const toggleId = `sidenote-toggle-${referenceId}`;

			context.replaceNode(node, {
				type: 'element',
				tagName: 'span',
				properties: { className: ['sidenote-anchor'] },
				children: [
					{
						type: 'element',
						tagName: 'label',
						properties: {
							className: ['sidenote-number'],
							htmlFor: toggleId,
							ariaLabel: `Toggle note ${number}`,
							dataNote: number,
						},
						children: [],
					},
					{
						type: 'element',
						tagName: 'input',
						properties: {
							className: ['margin-toggle'],
							id: toggleId,
							type: 'checkbox',
							ariaLabel: `Toggle note ${number}`,
						},
						children: [],
					},
					{
						type: 'element',
						tagName: 'span',
						properties: { className: ['sidenote'], dataNote: number, role: 'note' },
						children: noteContent,
					},
				],
			});
		},
	},
});
