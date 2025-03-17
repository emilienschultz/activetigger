import { FC, useState } from 'react';
import { FaRegTrashAlt } from 'react-icons/fa';
import { useDeleteFile, useGetFiles, useUploadFile } from '../core/api';

interface FileUploadProps {}

// to do :
// get available files
// delete existing file

export const FilesManagement: FC<FileUploadProps> = () => {
  const [file, setFile] = useState<File | null>(null);
  const [manageMenu, setManageMenu] = useState(false);
  const uploadFile = useUploadFile();
  const { files, reFetchFiles } = useGetFiles();
  const deleteFile = useDeleteFile(reFetchFiles);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files && event.target.files.length > 0) {
      setFile(event.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    uploadFile(file);
  };

  const formData = new FormData();
  formData.append('file', file || '');

  return (
    <div>
      <div className="explanations">Upload tabular file to use</div>
      <div className="d-flex align-items-center">
        <input type="file" onChange={handleFileChange} className="form-control" />
        <button onClick={handleUpload} disabled={!file} className="btn btn-primary mx-2">
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
