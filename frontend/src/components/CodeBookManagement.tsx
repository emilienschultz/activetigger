import MDEditor from '@uiw/react-md-editor';
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
    console.log('save codebook', time);
    reFetchCodebook();
    console.log('reFetchCodebook');
    console.log(codebook, time);
  };
  console.log(codebook, time);

  return (
    <div className="container mt-3">
      <MDEditor
        preview="preview"
        value={modifiedCodebook}
        onChange={setModifiedCodebook}
        previewOptions={{
          rehypePlugins: [[rehypeSanitize]],
        }}
      />
      <button className="btn btn-primary mt-3" onClick={saveCodebook}>
        Save modifications
      </button>
    </div>
  );
};
