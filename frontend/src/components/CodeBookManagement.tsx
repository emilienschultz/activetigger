import MDEditor from '@uiw/react-md-editor';
import { FC, useEffect, useState } from 'react';
import rehypeSanitize from 'rehype-sanitize';
import { usePostSchemeCodebook } from '../core/api';

interface CodebookManagementProps {
  projectName: string | null;
  currentScheme: string | null;
  codebook: string | null;
  time: string | null;
  reFetchCodebook: () => void;
}

/**
 * TODO : there is a need to manage the multiuser
 */

export const CodebookManagement: FC<CodebookManagementProps> = ({
  projectName,
  currentScheme,
  codebook,
  time,
  reFetchCodebook,
}) => {
  const { postCodebook } = usePostSchemeCodebook(projectName || null, currentScheme || null);

  const [modifiedCodebook, setModifiedCodebook] = useState<string | undefined>(undefined);
  //  const [lastModified, setLastModified] = useState<string | undefined | null>(undefined);

  // update the text zone once (if undefined)
  useEffect(() => {
    if (codebook && modifiedCodebook === undefined) {
      setModifiedCodebook(codebook);
    }
  }, [codebook, modifiedCodebook, time]);

  const saveCodebook = async () => {
    postCodebook(modifiedCodebook || '', time || '');
    reFetchCodebook();
  };

  return (
    <div>
      <MDEditor
        value={modifiedCodebook || ''}
        onChange={setModifiedCodebook}
        previewOptions={{
          rehypePlugins: [[rehypeSanitize]],
        }}
      />
      <button className="btn btn-secondary btn-sm mt-3" onClick={saveCodebook}>
        Save
      </button>
    </div>
  );
};
