// @ts-check
import { satteri } from '@astrojs/markdown-satteri';
import { defineConfig } from 'astro/config';
import { collectFootnotes, renderSidenotes } from './src/markdown/sidenotes.mjs';

// https://astro.build/config
export default defineConfig({
	site: process.env.SITE,
	base: process.env.BASE_PATH || '/',
	markdown: {
		processor: satteri({
			hastPlugins: [collectFootnotes, renderSidenotes],
		}),
	},
});
