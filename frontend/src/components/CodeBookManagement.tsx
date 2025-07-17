import MDEditor from '@uiw/react-md-editor';
import { marked } from 'marked';
import { FC, useEffect, useState } from 'react';
import rehypeSanitize from 'rehype-sanitize';
import { useGetSchemeCodebook, usePostSchemeCodebook } from '../core/api';

interface CodebookManagementProps {
  projectName: string | null;
  currentScheme: string | null;
}

/**
 * TODO : there is a need to manage the multiuser
 */

export const CodebookManagement: FC<CodebookManagementProps> = ({ projectName, currentScheme }) => {
  const { postCodebook } = usePostSchemeCodebook(projectName || null, currentScheme || null);
  const { codebook, time, reFetchCodebook } = useGetSchemeCodebook(
    projectName || null,
    currentScheme || null,
  );
  const [modifiedCodebook, setModifiedCodebook] = useState<string | undefined>(undefined);
  //  const [lastModified, setLastModified] = useState<string | undefined | null>(undefined);

  // update the text zone once (if undefined)
  useEffect(() => {
    if (codebook && modifiedCodebook === undefined) {
      setModifiedCodebook(codebook);
      //      setLastModified(time);
    }
  }, [codebook, modifiedCodebook, time]);

  const saveCodebook = async () => {
    postCodebook(modifiedCodebook || '', time || '');
    reFetchCodebook();
  };

  // Downoad
  const downloadMarkdown = () => {
    const blob = new Blob([modifiedCodebook || ''], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'codebook.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const openAsHTML = () => {
    const htmlContent = `
    <html>
      <head>
        <title>Codebook</title>
        <meta charset="UTF-8" />
        <style>
          body { font-family: sans-serif; padding: 2em; }
        </style>
      </head>
      <body>
        <div>${marked.parse(modifiedCodebook || '')}</div>
      </body>
    </html>
  `;
    const blob = new Blob([htmlContent], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    window.open(url, '_blank');
  };

  return (
    <div className="container">
      <div className="explanations">Keep track of the tagging rules</div>
      <MDEditor
        value={modifiedCodebook || ''}
        onChange={setModifiedCodebook}
        previewOptions={{
          rehypePlugins: [[rehypeSanitize]],
        }}
      />
      <button className="btn btn-secondary mt-3" onClick={saveCodebook}>
        Save
      </button>
      <button className="btn btn-primary mt-3 ms-2" onClick={openAsHTML}>
        Open
      </button>
      <button className="btn btn-primary mt-3 ms-2" onClick={downloadMarkdown}>
        Download
      </button>
    </div>
  );
};
