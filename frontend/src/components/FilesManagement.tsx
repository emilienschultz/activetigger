/**
 * THIS COMPONENT IS NOT USED FOR THE MOMENT
 */

import { FC, useState } from 'react';
import { FaRegTrashAlt } from 'react-icons/fa';
import { useDeleteFile, useGetFiles } from '../core/api';
import { useNotifications } from '../core/notifications';

interface FileUploadProps {}

//NOTE: Unused??
export const FilesManagement: FC<FileUploadProps> = () => {
  const MAX_FILE_SIZE = 500 * 1024 * 1024; // 500 MB in bytes
  const { notify } = useNotifications();
  const [file, setFile] = useState<File | null>(null);
  const [manageMenu, setManageMenu] = useState(false);
  // const uploadFile = useUploadFile();
  const { files, reFetchFiles } = useGetFiles();
  const deleteFile = useDeleteFile(reFetchFiles);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      if (event.target.files[0].size > MAX_FILE_SIZE) {
        notify({ type: 'error', message: 'File size is too big. Limit is 500 Mo.' });
        return;
      }
      setFile(event.target.files[0]);
    }
  };

  const formData = new FormData();
  formData.append('file', file || '');

  return (
    <div>
      <div className="explanations">Upload tabular file to use</div>
      <div className="d-flex align-items-center">
        <input type="file" onChange={handleFileChange} className="form-control" />
        <button
          onClick={async () => {
            if (!file) return;
            console.log('Uploading file', file);
            //await uploadFile(file);
          }}
          disabled={!file}
          className="btn btn-primary mx-2"
        >
          Upload
        </button>
        <button className="btn btn-primary">
          <span onClick={() => setManageMenu(!manageMenu)}>Manage</span>
        </button>
      </div>
      {manageMenu && (
        <div>
          <div className="explanations">Files available</div>
          <div>
            {(files || []).map((file) => (
              <div key={file} className="d-flex align-items-center">
                <button onClick={() => deleteFile(file)} className="btn">
                  <FaRegTrashAlt size={20} className="m-2" />
                </button>
                <span>{file}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
