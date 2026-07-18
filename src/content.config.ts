import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

const blogs = defineCollection({
	loader: glob({ base: './src/content/blogs', pattern: '**/*.{md,mdx}' }),
	schema: z.object({
		title: z.string(),
		description: z.string(),
		published: z.coerce.date(),
		author: z.string(),
		readTime: z.string(),
		draft: z.boolean().optional().default(false),
	}),
});

export const collections = { blogs };
