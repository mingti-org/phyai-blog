const basePath = `${import.meta.env.BASE_URL.replace(/\/+$/, '')}/`;

export const sitePath = (path = '') => `${basePath}${path.replace(/^\/+/, '')}`;
