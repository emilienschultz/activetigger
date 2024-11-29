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

  const { codebook } = useGetSchemeCodebook(projectName || null, currentScheme || null);
  const [modifiedCodebook, setModifiedCodebook] = useState<string | undefined>(undefined);
  // const [previousCodebook, setPreviousCodebook] = useState<string | undefined>(undefined);
  // const [hasChanged, setHasChanged] = useState<boolean>(false);

  // update the text zone once (if undefined)
  useEffect(() => {
    if (codebook && modifiedCodebook === undefined) {
      setModifiedCodebook(codebook);
      //      setPreviousCodebook(codebook);
    }
  }, [codebook, modifiedCodebook]);

  const saveCodebook = async () => {
    postCodebook(modifiedCodebook || '');
  };

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
      {/* {hasChanged && (
        <div className="alert alert-warning mt-3">
          Someone modified the online version of the codebook.
        </div>
      )} */}
    </div>
  );
};
