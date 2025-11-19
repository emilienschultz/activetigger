import MDEditor from '@uiw/react-md-editor';
import { Dispatch, FC, useEffect, SetStateAction } from 'react';
import rehypeSanitize from 'rehype-sanitize';

interface CodebookManagementProps {
  codebook: string | null;
  time: string | null;
  modifiedCodebook: string | undefined;
  setModifiedCodebook: Dispatch<SetStateAction<string | undefined>>;
  saveCodebook: () => Promise<void>;
}

/**
 * TODO : there is a need to manage the multiuser
 */

export const CodebookManagement: FC<CodebookManagementProps> = ({
  codebook,
  time,
  modifiedCodebook,
  setModifiedCodebook,
  saveCodebook,
}) => {
  // update the text zone once (if undefined)
  useEffect(() => {
    if (codebook && modifiedCodebook === undefined) {
      setModifiedCodebook(codebook);
    }
  }, [codebook, modifiedCodebook, time]);

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
